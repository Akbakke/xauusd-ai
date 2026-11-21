"""One native M5/closed-MTF TRAIN/development-VAL TA cache; no learning."""
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
from pyarrow import types as arrow_types
from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as owner
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
from gx1.features.volatility_squeeze_state_v1 import load_volatility_squeeze_artifact_manifest

REPO=owner.REPO;POLICY=REPO/'NEXT_RUN_POLICY.json'
STAGE='native_m5_ta_five_year_cache_20261120_001'
PLAN=REPO/'configs/research/NATIVE_M5_TA_FIVE_YEAR_CACHE_20261120_001.json'
ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_FIVE_YEAR_CACHE_20261120_001')
START_NS=1609459200000000000  # 2021-01-01T00:00:00Z
END_NS=1782864000000000000    # 2026-07-01T00:00:00Z
INPUT_REFS={
    'cache_qualification':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003/CACHE_QUALIFICATION.json',
        'sha256':'5c35042f655709e5b7d7366f5e761bb5988b8cd0173dd0eb4ce5b0833bd07e27'},
    'cache_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003/RESULT.json',
        'sha256':'1caaee7e6a6267b5af8aea453556afb272f7512bfbd7f596aa1386ce82b4c2fd'},
    'cache_terminal':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_HISTORY_EXPANSION_20261120_003/TERMINAL.json',
        'sha256':'4252b29e0c3a6c79941471d1c755c1da69547cee82788d84c68dc1bebbdc3d1c'}}
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


def research_read_end_numeric_identity(source):
    """Normalize only the qualified panel-row exposure predicate, never math."""
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='load_oanda_panel_d1']
    expected=ast.parse("pd.read_parquet(path, columns=['time','open','high','low','close'], "
        "filters=[('time','<',sealed_period_read_end(spec))])",mode='eval').body
    calls=[n for n in ast.walk(functions[0])if isinstance(n,ast.Call)
        and ast.dump(n.func)==ast.dump(expected.func)]if len(functions)==1 else []
    need(len(calls)==1 and ast.dump(calls[0])==ast.dump(expected),'EXACT_PANEL_READ_END_PREDICATE')
    calls[0].keywords=[kw for kw in calls[0].keywords if kw.arg!='filters']
    return research_import_numeric_identity(ast.unparse(tree))


def research_import_bridge(old_plan):
    original=subprocess.check_output(['git','show',old_plan['source_base_commit']+':'+RESEARCH_SOURCE_PATH],cwd=REPO)
    original_sha=hashlib.sha256(original).hexdigest()
    need(original_sha==old_plan['source_owners'][RESEARCH_SOURCE_PATH],'ORIGINAL_RESEARCH_SOURCE')
    before=research_import_numeric_identity(original.decode())
    after=research_read_end_numeric_identity((REPO/RESEARCH_SOURCE_PATH).read_text())
    need(before==after,'RESEARCH_NON_IMPORT_AST_CHANGED')
    return {'schema_version':'gx1_native_m5_macro_research_qualified_read_end_bridge_v1',
        'source_owner':RESEARCH_SOURCE_PATH,'original_sha256':original_sha,'current_sha256':sha(REPO/RESEARCH_SOURCE_PATH),
        'import_only_normalized_ast_sha256':after,'relocated_import_sources':list(RESEARCH_LAZY_IMPORTS),
        'qualified_panel_reader_change':'load_oanda_panel_d1.pd.read_parquet strict time<sealed_period_read_end(spec)',
        'all_other_module_statements_and_macro_function_bodies_unchanged':True}






def phase(name):
    return publish_artifact('PHASE_'+name+'.json',{'schema_version':'gx1_native_ta_cache_phase_v1',
        'stage':STAGE,'phase':name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'ru_maxrss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})


def metadata(inputs):
    need(type(inputs)is dict and inputs==INPUT_REFS,'EXACT_QUALIFIED_PARENT')
    qualification,result,terminal=(read(inputs[k])for k in ('cache_qualification','cache_result','cache_terminal'))
    parent=read(result['plan']);claim=read(result['claim'])
    need(qualification['schema_version']=='gx1_native_m5_six_year_TA_input_qualification_v1'
        and qualification['result']==terminal['result']==inputs['cache_result']
        and qualification['terminal']==inputs['cache_terminal']
        and result['stage']==terminal['stage']==parent['stage']==claim['stage']=='native_m5_ta_history_expansion_20261120_003'
        and result['status']=='COMPLETE_NATIVE_M5_TA_FEATURE_CACHE'and terminal['status']=='COMPLETE'
        and terminal['exit_code']==0 and terminal['plan']==claim['plan']==result['plan']
        and terminal['claim']==result['claim']and result['source_unchanged']is terminal['source_unchanged']is True
        and qualification['all_issue_rows_ready']is qualification['all61_macro_fields_observed_on_all_issue_rows']is True
        and all(qualification[k]is False for k in ('M1_used','GC_used','test_data_used'))
        and result['test_data_used']is terminal['test_data_used']is False,'COMPLETE_QUALIFIED_CACHE_PARENT')
    snapshot_ref=qualification['snapshot'];archive_ref=qualification['archive'];saved=read(snapshot_ref)
    need(snapshot_ref==result['report']['snapshot']and archive_ref==result['report']['archive']==saved['archive']
        and saved['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'and len(saved['arrays'])==27
        and saved['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256']is True,'QUALIFIED_REFERENCE_CACHE')
    previous=parent['metadata'];plan_ref=previous['dependencies']['qualified_plan'];plan=read(plan_ref)
    parameter_inputs=previous['parameter_inputs'];source_ref=parameter_inputs['native_source_plan'];source=read(source_ref)
    facts_ref=plan['inputs']['target_result'];facts_terminal_ref=plan['inputs']['target_terminal']
    facts,facts_terminal=read(facts_ref),read(facts_terminal_ref)
    need(facts['plan']==facts_terminal['plan']==source_ref and facts_terminal['result']==facts_ref
        and facts_terminal['status']=='COMPLETE'and facts_terminal['exit_code']==0
        and facts['source_unchanged']is facts_terminal['source_unchanged']is True
        and facts['source_spec']==source['source_spec']and facts['test_data_used']is facts_terminal['test_data_used']is False,
        'COMPLETE_NATIVE_SOURCE_PREFIX')
    original=source['source_spec']
    need(original['cutoff_ns']==END_NS and original['issue_start_ns']<START_NS
        and len(original['parts'])==18 and original['manifest']==saved['surfaces_metadata']['source_binding']['native_m5_manifest'],
        'FULL_NATIVE_PREFIX_SCOPE')
    spec={key:original[key]for key in ('manifest','parts','cutoff_ns')}
    parameters,_=owner.parameter_artifact_metadata(parameter_inputs,previous['geometry'])
    contract=features.native_m5_feature_provider_contract()
    need(parameters==previous['parameter_reuse']and contract==saved['surfaces_metadata']['provider_contract']
        and saved['surfaces_metadata']['parameter_available_ns']<=START_NS,'FROZEN_NATIVE_PARAMETERS')
    macro_ref=previous['dependencies']['macro_source_plan'];macro_dependencies=owner.macro_metadata(macro_ref)
    dependencies={**macro_dependencies,**parameters['parameter_dependencies'],
        'qualified_cache_plan':result['plan'],'qualified_cache_claim':result['claim'],
        'qualified_snapshot':snapshot_ref,'qualified_archive':archive_ref,'qualified_origin_plan':plan_ref,
        'native_source_plan':source_ref,'native_facts_result':facts_ref,'native_facts_terminal':facts_terminal_ref,
        'registry':parameters['registry_artifact'],'squeeze_manifest':parameters['squeeze_manifest'],
        'native_manifest':spec['manifest'],'macro_source_plan':macro_ref,
        'squeeze_fit_plan':parameter_inputs['squeeze_fit_plan'],'registry_fit_plan':parameter_inputs['registry_fit_plan']}
    dependencies.update({'native_part_'+str(i):{'path':part['path'],'sha256':part['sha256']}for i,part in enumerate(spec['parts'])})
    source_metadata={part['path']:owner.data.part_metadata(part)for part in spec['parts']}
    need(source_metadata==source['source_metadata'],'FULL_PREFIX_METADATA_UNCHANGED')
    source_arrow_schema={}
    for part in spec['parts']:
        schema=owner.pq.ParquetFile(part['path']).schema_arrow
        columns=['time',*features.RAW_COLUMNS]
        need(all(name in schema.names for name in columns),'SOURCE_ARROW_COLUMNS')
        clock=schema.field('time').type
        need(arrow_types.is_timestamp(clock)and clock.tz=='UTC'
            and all(arrow_types.is_floating(schema.field(name).type)or arrow_types.is_integer(schema.field(name).type)
                    for name in features.RAW_COLUMNS),'SOURCE_ARROW_TYPES')
        source_arrow_schema[part['path']]={name:str(schema.field(name).type)for name in columns}

    expected_years={str(y):facts['report']['issue_rows_by_year'][str(y)]for y in range(2021,2027)}
    no_torch();numeric_owners,alias_bridge=numeric_owner_identities(plan)
    return {'dependencies':dependencies,'feature_source_spec':spec,'feature_source_metadata':source_metadata,
        'source_arrow_schema':source_arrow_schema,'parameter_inputs':parameter_inputs,'geometry':previous['geometry'],'parameter_reuse':parameters,
        'feature_contract':contract,'expected_source_rows':facts['report']['source_rows'],
        'expected_issue_rows_by_year':expected_years,'reference_snapshot':snapshot_ref,'reference_archive':archive_ref,
        'qualified_parent':inputs,'numeric_owner_identities':numeric_owners,'alias_layout_bridge':alias_bridge,
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


def _decode_snapshot_surfaces(snapshot_ref):
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
    return surfaces


def load_snapshot(snapshot_ref):
    return features.NativeM5FeatureProvider(_decode_snapshot_surfaces(snapshot_ref))


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
        def shared(current,previous,reason):
            ids,left,right=np.intersect1d(current,previous,assume_unique=True,return_indices=True)
            need(len(ids)>0,reason)
            return ids,left,right
        local,at,before=shared(provider.retained_source_rows,get('retained_source_rows'),'OVERLAP_GLOBAL_M5_IDS')
        for key in ('signal','ctx_cont','ctx_cat','observed'):
            equal(key,provider.surfaces[key][at],get(key)[before])
        equal('retained_source_rows',provider.retained_source_rows[at],get('retained_source_rows')[before])
        issues,present,previous=shared(provider.surfaces['retained_issue_rows'],get('retained_issue_rows'),'OVERLAP_ISSUE_IDS')
        equal('retained_issue_rows',provider.surfaces['retained_issue_rows'][present],get('retained_issue_rows')[previous])
        original_starts=get('bar_start_ns')
        need(len(provider.bar_start_ns)>=len(original_starts),'OVERLAP_FULL_SOURCE_PREFIX')
        equal('bar_start_ns',provider.bar_start_ns[:len(original_starts)],original_starts)
        need(provider.input_ready[issues].all(),'OVERLAP_PREVIOUSLY_READY')
        for tf in features.TIMEFRAMES:
            frame=provider.surfaces['multi_tf_sources'][tf]
            ids,positions,previous=shared(frame.attrs['owner_row_indices'],get('tf_'+tf+'_owner_row_indices'),
                'OVERLAP_GLOBAL_TF_IDS:'+tf)
            for key in ('owner_row_indices','ts_int64','feats_np','model_native_mtf_scalars_np_v4'):
                equal('tf_'+tf+'_'+key,frame.attrs[key][positions],get('tf_'+tf+'_'+key)[previous])
    need(len(compared)==27 and ref(archive_ref['path'])==archive_ref,'OVERLAP_BOOKEND')
    return {'array_count':27,'exact_dtype_shape_bytes_and_global_owner_clocks':True,'arrays':compared,
        'reference_snapshot':snapshot_ref,'reference_archive':archive_ref,'reference_issues':len(issues),
        'shared_local_rows':len(local),'original_full_bar_axis_is_exact_prefix':True,
        'overlap_first_issue_ns':int(provider.issue_ns[issues[0]]),'overlap_last_issue_ns':int(provider.issue_ns[issues[-1]])}


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
    need(binding['source_spec']==meta['feature_source_spec']
        and binding['native_m5_manifest']==meta['feature_source_spec']['manifest']
        and binding['rows']==meta['expected_source_rows'],'FULL_PREFIX_SOURCE_IDENTITY')
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
    annual=yearly_readiness(provider)
    need({year:value['rows']for year,value in annual.items()}==meta['expected_issue_rows_by_year'],'ISSUE_YEAR_COUNTS')
    parity=overlap_parity(provider,meta['reference_snapshot'],meta['reference_archive'])
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
    if '--cot-normalization' in sys.argv[1:]:
        return cot_norm_main(sys.argv[1:])
    if '--cot-publication-correction' in sys.argv[1:]:
        return cot_main(sys.argv[1:])
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out-dir',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();no_torch();install_native_m5_direction_io_firewall()
    need(args.plan==str(PLAN)and args.out_dir==str(ROOT),'CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_TA_CACHE_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'cached_values_decoded':0,'Torch_imported':False,**COUNTS}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


# Calendar-only repair reuses authenticated physical TA arrays; it never rebuilds TA.
COT_STAGE='native_m5_ta_cot_clock_correction_20261121_001'
COT_PARENT='native_future_state_train_readiness_20261121_001'
COT_GATE='cot_cache_correction_001'
COT_PLAN=REPO/'configs/research/NATIVE_M5_TA_COT_CLOCK_CORRECTION_20261121_001.json'
COT_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_COT_CLOCK_CORRECTION_20261121_001')
COT_MODULE='gx1.audit.build_native_m5_ta_feature_cache_v1'
COT_LEGACY_COMMIT='3a0d67b2aacdfb2bd1155119d80d320fb2894e0d'
COT_LEGACY_SPEC={'path':'configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json',
    'sha256':'b8e2c01ddc5c1941dd509f1c3507a4211b2d0f7e3820f64c7607e95a0916d6b3'}
COT_LEGACY_CLOCK={'path':'configs/research/CFTC_COT_PUBLICATION_CLOCK_20261018_001.json',
    'sha256':'64b6d2e0566f977b38f98e02848e4fdff7bb0b7a32c31cd50e18b47d7f91e0d2'}
COT_INPUTS={
    'cache_result':{'path':str(ROOT/'RESULT.json'),'sha256':'b2f3b4f16262bd6e09f4186810d37312751751b9f19740e0f1572a58faaab39f'},
    'cache_terminal':{'path':str(ROOT/'TERMINAL.json'),'sha256':'fd50aa03c4ee5f154887cb0cd375c19982e9c7f7d528db3f0311bddc270cdff0'},
    'cache_qualification':{'path':str(ROOT/'CACHE_QUALIFICATION.json'),'sha256':'f8b1d7fa8745276000a276420851c69656a7788d135c22123541d7c2a28ba8ab'}}
COT_FIELDS=('cot_noncommercial_net_over_oi','cot_change4reports',
    'cot_open_interest_log_change4reports','cot_commercial_net_over_oi')
COT_EXECUTION={'class':'producer','device':'cpu','memory':'12G','swap':'512M','tasks_max':64,
    'numerical_threads':1,'cpu_affinity':list(range(8)), 'internal_seconds':870,
    'outer_seconds':900,'kill_after_seconds':10,'external_wall_timeout_enforced':True}
COT_COUNTS={**COUNTS,'TA_rebuilds':0,'macro_materializations':1,
    'snapshot_publications':1,'snapshot_reloads':1,'legacy_arrays_decoded':27,'new_arrays_published':27}
COT_SOURCE_PATHS=tuple(dict.fromkeys((*SOURCE_PATHS,*owner.NUMERIC_OWNER_PATHS,
    'gx1/contracts/native_m5_direction_io_firewall_v1.py')))


def cot_binding(binding, *, json_only=False):
    need(type(binding)is dict and set(binding)=={'path','sha256'}
        and type(binding['path'])is str and Path(binding['path']).is_absolute()
        and '..'not in Path(binding['path']).parts and type(binding['sha256'])is str
        and len(binding['sha256'])==64 and all(c in '0123456789abcdef'for c in binding['sha256']),
        'COT_REFERENCE')
    need(not json_only or Path(binding['path']).suffix=='.json','COT_JSON_ONLY')
    return binding


def cot_read(binding):
    return read(cot_binding(binding,json_only=True))


def cot_json_ref(path):
    need(Path(path).suffix=='.json','COT_JSON_ONLY')
    return cot_binding(ref(path),json_only=True)


def cot_git_json(binding):
    need(binding in (COT_LEGACY_SPEC,COT_LEGACY_CLOCK),'COT_LEGACY_JSON_SCOPE')
    raw=subprocess.check_output(['git','show',COT_LEGACY_COMMIT+':'+binding['path']],cwd=REPO)
    need(hashlib.sha256(raw).hexdigest()==binding['sha256'],'COT_LEGACY_JSON_HASH')
    return json.loads(raw)


def cot_legacy_provider(current):
    legacy=json.loads(json.dumps(current));macro_contract=legacy['macro_contract']
    macro_contract['source_spec']=dict(COT_LEGACY_SPEC)
    macro_contract['contract_sha256']=features._sha({k:v for k,v in macro_contract.items()if k!='contract_sha256'})
    return features._seal({k:v for k,v in legacy.items()if k!='contract_sha256'})


def cot_spec_transition(legacy,current,old_clock,new_clock):
    expected=json.loads(json.dumps(legacy));expected['cot']['publication_clock']=current['cot']['publication_clock']
    need(current==expected and legacy['cot']['publication_clock']==COT_LEGACY_CLOCK
        and current['cot']['publication_clock']['path']==COT_LEGACY_CLOCK['path']
        and current['cot']['publication_clock']['sha256']!=COT_LEGACY_CLOCK['sha256'],
        'COT_ONLY_CLOCK_SOURCE_CHANGE')
    need(old_clock['schema_version']==new_clock['schema_version']=='gx1_cftc_cot_delayed_publication_clock_v1'
        and type(new_clock['rows'])is list and len(new_clock['rows'])>len(old_clock['rows'])
        and all(row in new_clock['rows']for row in old_clock['rows']), 'COT_CLOCK_EXTENSION')
    dates=[row['as_of']for row in new_clock['rows']]
    need(len(set(dates))==len(dates),'COT_CLOCK_DUPLICATE')


def cot_macro_dependencies(spec):
    dependencies={}
    refs=[spec['cot']['result'],spec['oanda_panel']['result'],
        *(spec['drivers'][key]for key in ('treasury','vix','gld'))]
    for binding in refs:
        result=cot_read(binding);dependencies[binding['path']]=binding
        need(type(result.get('records'))is list and result['records'],'COT_MACRO_RECORDS')
        for record in result['records']:
            need(record['status']=='RETRIEVED_NOT_ADMITTED','COT_MACRO_INCOMPLETE')
            pair=('parquet_path','parquet_sha256')if 'parquet_path'in record else('raw_path','raw_sha256')
            value=cot_binding({'path':record[pair[0]],'sha256':record[pair[1]]})
            need(value['path']not in dependencies or dependencies[value['path']]==value,'COT_SPLIT_DEPENDENCY')
            dependencies[value['path']]=value
    return dependencies


def cot_metadata():
    from gx1.features import d1_macro_context_v1 as macro
    result,terminal,qualification=(cot_read(COT_INPUTS[k])for k in
        ('cache_result','cache_terminal','cache_qualification'))
    need(result['stage']==terminal['stage']==STAGE and result['status']=='COMPLETE_NATIVE_M5_TA_FEATURE_CACHE'
        and terminal['status']=='COMPLETE'and terminal['exit_code']==0
        and result['source_unchanged']is terminal['source_unchanged']is True
        and result['test_data_used']is terminal['test_data_used']is False
        and terminal['result']==qualification['result']==COT_INPUTS['cache_result']
        and qualification['terminal']==COT_INPUTS['cache_terminal']
        and terminal['plan']==result['plan'] and terminal['claim']==result['claim'], 'COT_PARENT_LINEAGE')
    parent=cot_read(result['plan']);claim=cot_read(result['claim'])
    need(parent['stage']==claim['stage']==STAGE and claim['plan']==result['plan']
        and qualification['schema_version']=='gx1_native_m5_five_year_TA_input_qualification_v1'
        and qualification['all_issue_rows_ready']is qualification['all61_macro_fields_observed_on_all_issue_rows']is True
        and all(qualification[k]is False for k in ('M1_used','GC_used','test_data_used')),'COT_PARENT_QUALIFIED')
    snapshot_ref=result['report']['snapshot'];snapshot=cot_read(snapshot_ref)
    archive=cot_binding(snapshot['archive'])
    need(snapshot_ref==qualification['snapshot']and archive==qualification['archive']==result['report']['archive']
        and snapshot['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'
        and len(snapshot['arrays'])==27 and snapshot['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256']is True,
        'COT_PARENT_SNAPSHOT')
    legacy_spec=cot_git_json(COT_LEGACY_SPEC);old_clock=cot_git_json(COT_LEGACY_CLOCK)
    spec=macro.load_d1_macro_context_source_spec(REPO)
    spec_ref=cot_json_ref(REPO/macro.D1_MACRO_CONTEXT_SOURCE_SPEC['path'])
    need(spec_ref['sha256']==macro.D1_MACRO_CONTEXT_SOURCE_SPEC['sha256'],'COT_CURRENT_SPEC')
    clock_ref=cot_json_ref(REPO/spec['cot']['publication_clock']['path'])
    need(clock_ref['sha256']==spec['cot']['publication_clock']['sha256'],'COT_CURRENT_CLOCK')
    clock=cot_read(clock_ref);cot_spec_transition(legacy_spec,spec,old_clock,clock)
    need(len(clock['rows'])==21,'COT_EXACT_CALENDAR_ROWS')
    contract=features.native_m5_feature_provider_contract();legacy_contract=cot_legacy_provider(contract)
    surfaces=snapshot['surfaces_metadata'];binding=surfaces['macro_binding']
    need(surfaces['provider_contract']==legacy_contract and binding['contract']==legacy_contract['macro_contract']
        and binding['sources']=={k:legacy_spec[k]for k in ('cot','oanda_panel','drivers')}
        and binding['binding_sha256']==features._sha({k:v for k,v in binding.items()if k!='binding_sha256'})
        and surfaces['axes']['macro_binding_sha256']==binding['binding_sha256'], 'COT_LEGACY_PROVIDER_TRANSITION')
    need(surfaces['axes']['issue_start_ns']==START_NS and surfaces['axes']['issue_end_ns']==END_NS
        and qualification['qualified_issue_rows']==389484 and qualification['retained_local_rows']==389579,
        'COT_RETAINED_SCOPE')
    dependencies=cot_macro_dependencies(spec)
    for value in [*COT_INPUTS.values(),result['plan'],result['claim'],snapshot_ref,spec_ref,clock_ref,
            surfaces['source_binding']['native_m5_manifest']]:
        cot_binding(value,json_only=True)
        need(value['path']not in dependencies or dependencies[value['path']]==value,'COT_SPLIT_DEPENDENCY')
        cot_read(value);dependencies[value['path']]=value
    return {'dependencies':dependencies,'reference_snapshot':snapshot_ref,'reference_archive':archive,
        'legacy_commit':COT_LEGACY_COMMIT,'legacy_source_spec':COT_LEGACY_SPEC,'legacy_clock':COT_LEGACY_CLOCK,
        'legacy_provider_contract':legacy_contract,'feature_contract':contract,'current_source_spec':spec_ref,
        'current_clock':clock_ref,'current_clock_rows':clock['rows'],'expected_annual':qualification['annual'],
        'TRAIN_rows':354550,'development_rows':34934,'retained_local_rows':389579,
        'TRAIN_issue_ns_sha256':'c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea'}


def cot_build_plan(*,source_base_commit=None):
    metadata=cot_metadata()
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_ta_cot_clock_correction_plan_v1','stage':COT_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(COT_ROOT),'source_base_commit':head,
        'source_owners':{name:sha(REPO/name)for name in COT_SOURCE_PATHS},'inputs':COT_INPUTS,
        'metadata':metadata,'execution':COT_EXECUTION,'limits':COT_COUNTS,
        'training_authorized':False,'test_data_used':False,'GC_used':False,'M1_used':False,
        'normalization_or_model_state_emitted':False}


def cot_source_clean(plan):
    need(type(plan['source_owners'])is dict and set(plan['source_owners'])==set(COT_SOURCE_PATHS),'COT_SOURCE_SET')
    return all(sha(REPO/name)==plan['source_owners'][name]for name in COT_SOURCE_PATHS)


def cot_checked_inputs(plan,*,payloads):
    for binding in plan['metadata']['dependencies'].values():
        cot_binding(binding)
        if payloads:need(ref(binding['path'])==binding,'COT_DEPENDENCY_BOOKEND')
        elif Path(binding['path']).suffix=='.json':cot_read(binding)
    if payloads:
        binding=plan['metadata']['reference_archive']
        need(ref(binding['path'])==binding,'COT_ARCHIVE_BOOKEND')
    no_torch()


def cot_inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not COT_ROOT.exists()and not COT_ROOT.is_symlink(),'COT_CWD_OUTPUT')
    plan_ref=cot_binding({'path':str(COT_PLAN),'sha256':plan_sha},json_only=True);plan=cot_read(plan_ref)
    # Validate exact source inventory BEFORE any path in it may be hashed.
    need(type(plan.get('source_owners'))is dict and set(plan['source_owners'])==set(COT_SOURCE_PATHS),'COT_SOURCE_SET')
    need(plan==cot_build_plan(source_base_commit=plan['source_base_commit'])and cot_source_clean(plan),'COT_PLAN_SOURCE')
    policy_ref=cot_json_ref(POLICY);policy=cot_read(policy_ref)
    parent=policy.get(COT_PARENT,{});gate=parent.get(COT_GATE,{})
    need(all(policy.get(k)is False for k in owner.data.CLOSED_FLAGS)
        and parent.get('goal_status')=='ACTIVE'and parent.get('large_training_qualified')is False
        and gate.get('plan')==plan_ref and gate.get('output_root')==str(COT_ROOT)
        and gate.get('execution')==COT_EXECUTION and gate.get('limits')==COT_COUNTS
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed')is False and gate.get('training_authorized')is False
        and gate.get('test_authorized')is False,'COT_CLOSED_POLICY')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'COT_ONE_USE')
    cot_checked_inputs(plan,payloads=False);return plan,plan_ref,policy_ref


def cot_replace_surfaces(surfaces,frame,spec,metadata):
    from gx1.features import d1_macro_context_v1 as macro
    profile=features.owners.compile_full_ta_entry_execution_profile();fields=profile['physical_field_names']
    need(surfaces['provider_contract']==metadata['legacy_provider_contract']
        and cot_legacy_provider(metadata['feature_contract'])==metadata['legacy_provider_contract'], 'COT_LEGACY_PROVIDER_TRANSITION')
    rows=surfaces['retained_source_rows'];times=surfaces['bar_start_ns'][rows]+features.BAR_NS
    need(list(frame.columns)==['decision_time',*macro.D1_MACRO_CONTEXT_FIELDS]
        and np.array_equal(pd.DatetimeIndex(frame['decision_time']).as_unit('ns').asi8,times),'COT_MACRO_AXIS')
    signal=surfaces['signal'];ctx=surfaces['ctx_cont'];observed=surfaces['observed']
    need(np.array_equal(signal[:,list(profile['temporal_alias_signal_indices'])],
        ctx[:,list(profile['temporal_alias_ctx_cont_indices'])],equal_nan=True)
        and np.array_equal(observed,np.isfinite(signal)),'COT_LEGACY_ALIASES_MASK')
    macro_names=list(macro.D1_MACRO_CONTEXT_FIELDS);need(tuple(macro_names[:4])==COT_FIELDS and len(macro_names)==61,'COT_FIELD_IDENTITY')
    unchanged=[fields['signal'].index('ctx_cont.'+name)for name in macro_names if name not in COT_FIELDS]
    canonical=frame[[name for name in macro_names if name not in COT_FIELDS]].to_numpy(np.float32)
    need(digest(canonical)==digest(signal[:,unchanged]),'COT_NONCOT57_PARITY')
    new_signal=signal.copy();new_ctx=ctx.copy();changes={};si=[];ci=[]
    for name in COT_FIELDS:
        sj=fields['signal'].index('ctx_cont.'+name);cj=fields['ctx_cont'].index(name);si.append(sj);ci.append(cj)
        need((sj,cj)in zip(profile['temporal_alias_signal_indices'],profile['temporal_alias_ctx_cont_indices']), 'COT_ALIAS_IDENTITY')
        value=frame[name].to_numpy(np.float32);need(not np.isinf(value).any(),'COT_INFINITE')
        same=(signal[:,sj]==value)|(np.isnan(signal[:,sj])&np.isnan(value));changed=np.flatnonzero(~same)
        changes[name]={'changed_rows':len(changed),'first_changed_issue_ns':int(times[changed[0]])if len(changed)else None,
            'last_changed_issue_ns':int(times[changed[-1]])if len(changed)else None,
            'old_sha256':digest(signal[:,sj]),'new_sha256':digest(value)}
        new_signal[:,sj]=value;new_ctx[:,cj]=new_signal[:,sj]
    new_observed=np.isfinite(new_signal)
    for old,new,columns in ((signal,new_signal,si),(ctx,new_ctx,ci),(observed,new_observed,si)):
        keep=[j for j in range(old.shape[1])if j not in columns]
        need(digest(old[:,keep])==digest(new[:,keep]),'COT_ONLY_FOUR_COLUMNS')
    binding=macro.d1_macro_context_binding(spec,frame)
    need(binding['decision_clock_sha256']==surfaces['macro_binding']['decision_clock_sha256'],'COT_MACRO_CLOCK_PRESERVED')
    repaired={**surfaces,'signal':features._owned(new_signal,np.float32),'ctx_cont':features._owned(new_ctx,np.float32),
        'observed':features._owned(new_observed,np.bool_),'macro_binding':binding,
        'provider_contract':metadata['feature_contract']}
    axes={k:v for k,v in surfaces['axes'].items()if k!='contract_sha256'}
    axes['macro_binding_sha256']=binding['binding_sha256'];repaired['axes']=features._seal(axes)
    provider=features.NativeM5FeatureProvider(repaired)
    return provider,{'canonical_F64_macro_fields':61,'unchanged_F32_macro_fields':57,
        'changed_fields':changes,'alias_values_equal':True,'observed_equals_isfinite_signal':True,
        'unmodified_nonCOT_columns_byte_exact':True}


def cot_array_parity(surfaces,snapshot):
    actual={k:surfaces[k]for k in ('bar_start_ns','retained_source_rows','retained_issue_rows','signal','ctx_cont','ctx_cat','observed')}
    for tf,frame in surfaces['multi_tf_sources'].items():
        for name in ('ts_int64','feats_np','owner_row_indices','model_native_mtf_scalars_np_v4'):
            actual['tf_'+tf+'_'+name]=frame.attrs[name]
    preserved={}
    need(set(actual)==set(snapshot['arrays'])and len(actual)==27,'COT_ARRAY_INVENTORY')
    for name,value in actual.items():
        header=snapshot['arrays'][name]
        need(value.dtype.str==header['dtype']and list(value.shape)==header['shape'],'COT_ARRAY_HEADER:'+name)
        if name not in ('signal','ctx_cont','observed'):
            need(digest(value)==header['sha256'],'COT_PRESERVED_ARRAY:'+name);preserved[name]=header
    need(len(preserved)==24 and surfaces['field_lineage']==snapshot['surfaces_metadata']['field_lineage'],'COT_CLOCK_LINEAGE')
    return preserved


def cot_clock_witness(cot,*,calendar_rows,lag_days,read_end):
    """Check bound exceptions against reports already decoded by the canonical parser."""
    seal=pd.Timestamp(read_end);witness=[];excluded=0
    need(seal.tzinfo is not None and type(lag_days)is int and lag_days>=0,'COT_WITNESS_CLOCK')
    dates=[row['as_of']for row in calendar_rows]
    need(len(set(dates))==len(dates),'COT_WITNESS_DUPLICATE_CALENDAR')
    for row in calendar_rows:
        as_of=pd.Timestamp(row['as_of'],tz='UTC')
        if as_of>=seal:
            excluded+=1;continue
        matched=cot.loc[cot['as_of']==as_of]
        need(len(matched)==1,'COT_WITNESS_REPORT_EXACTLY_ONCE:'+row['as_of'])
        actual=pd.Timestamp(matched['available_at'].iloc[0])
        lag=as_of+pd.Timedelta(days=lag_days)
        official=(pd.Timestamp(row['publication_date_ny'])+pd.Timedelta(days=1)).tz_localize(
            'America/New_York').tz_convert('UTC')
        expected=max(lag,official)
        need(actual.tzinfo is not None and actual.value==expected.value,'COT_WITNESS_AVAILABLE_AT:'+row['as_of'])
        actual_as_of=pd.Timestamp(matched['as_of'].iloc[0])
        pair=np.asarray([actual_as_of.value,actual.value],dtype='<i8')
        witness.append({'as_of':row['as_of'],'publication_date_ny':row['publication_date_ny'],
            'actual_as_of_ns':int(actual_as_of.value),'lag_available_ns':int(lag.value),
            'official_publication_day_end_ns':int(official.value),'expected_available_ns':int(expected.value),
            'actual_available_ns':int(actual.value),'actual_as_of_available_at_sha256':digest(pair)})
    pairs=np.asarray([[r['actual_as_of_ns'],r['actual_available_ns']]for r in witness],dtype='<i8').reshape(-1,2)
    return {'calendar_rows':len(calendar_rows),'matched_rows':len(witness),'excluded_at_or_after_seal':excluded,
        'all_declared_preseal_rows_matched_exactly_once':True,'all_available_at_exact_max_lag_official_day_end':True,
        'rows':witness,'actual_as_of_available_at_sha256':digest(pairs),'canonical_rows_sha256':features._sha(witness)}


def cot_produce(plan):
    from gx1.features import d1_macro_context_v1 as macro
    meta=plan['metadata'];counts={k:0 for k in COT_COUNTS};cot_checked_inputs(plan,payloads=True)
    snapshot=cot_read(meta['reference_snapshot']);surfaces=_decode_snapshot_surfaces(meta['reference_snapshot'])
    counts['legacy_arrays_decoded']=27
    need(surfaces['provider_contract']==meta['legacy_provider_contract'],'COT_LEGACY_PROVIDER_TRANSITION')
    spec=macro.load_d1_macro_context_source_spec(REPO)
    times=pd.to_datetime(surfaces['bar_start_ns'][surfaces['retained_source_rows']]+features.BAR_NS,utc=True)
    components=macro.d1_macro_context_components(spec)
    clock_witness=cot_clock_witness(components['cot'],calendar_rows=meta['current_clock_rows'],
        lag_days=spec['cot']['availability_lag_days'],read_end=components['read_end'])
    need(clock_witness['matched_rows']==clock_witness['calendar_rows']==21
        and clock_witness['excluded_at_or_after_seal']==0,'COT_ALL21_CLOCK_ROWS_MATCHED')
    frame=macro.build_d1_macro_context(times,components);counts['macro_materializations']+=1
    del components
    provider,changes=cot_replace_surfaces(surfaces,frame,spec,meta)
    preserved=cot_array_parity(provider.surfaces,snapshot)
    annual=yearly_readiness(provider);rows=provider.surfaces['retained_issue_rows'];issues=provider.issue_ns[rows]
    train=rows[issues<1767225600000000000];development=rows[issues>=1767225600000000000]
    need(len(train)==meta['TRAIN_rows']==354550 and len(development)==meta['development_rows']==34934
        and len(provider.retained_source_rows)==meta['retained_local_rows']==389579
        and digest(provider.issue_ns[train])==meta['TRAIN_issue_ns_sha256']
        and provider.input_ready[rows].all(),'COT_FIXED_INPUT_COHORTS')
    need(set(annual)==set(meta['expected_annual'])and all(annual[y]['rows']==meta['expected_annual'][y]['rows']
        and annual[y]['macro_current_all61_observed_rows']==annual[y]['rows']for y in annual),'COT_ANNUAL_COVERAGE')
    new_snapshot=owner.publish_feature_snapshot(provider,publish_artifact=cot_publish);counts['snapshot_publications']+=1
    counts['new_arrays_published']=27;new_header=cot_read(new_snapshot)
    expected_axes=provider.surfaces['axes'];expected_ready=digest(provider.input_ready)
    del provider,surfaces,frame
    import gc
    gc.collect()
    provider=load_snapshot(new_snapshot);counts['snapshot_reloads']+=1
    need(provider.surfaces['axes']==expected_axes and digest(provider.input_ready)==expected_ready
        and yearly_readiness(provider)==annual and counts==COT_COUNTS,'COT_RELOAD_COUNTS_READINESS')
    cot_array_parity(provider.surfaces,snapshot);no_torch()
    return {'snapshot':new_snapshot,'archive':new_header['archive'],'reference_snapshot':meta['reference_snapshot'],
        'reference_archive':meta['reference_archive'],'source_axes':provider.surfaces['axes'],
        'input_provider_contract':provider.surfaces['provider_contract'],'macro_binding':provider.surfaces['macro_binding'],
        'COT_correction':changes,'COT_clock_witness':clock_witness,
        'unchanged_arrays':preserved,'yearly_readiness':annual,
        'TRAIN_rows':len(train),'development_rows':len(development),'TRAIN_issue_ns_sha256':meta['TRAIN_issue_ns_sha256'],
        'rows':len(rows),'retained_local_rows':len(provider.retained_source_rows),'actual_counts':counts,
        'Torch_imported':False,'GC_used':False,'M1_used':False,'test_data_used':False,
        'normalization_refit_required':True,'initialbaseline_qualified':False}


def cot_publish(name,value):
    need(name in ('CLAIM.json','FEATURE_INPUTS.npz','FEATURE_INPUTS.json','RESULT.json','TERMINAL.json'),'COT_OUTPUT_NAME')
    raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    path=COT_ROOT/name;_publish_bytes(path,raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def cot_timeout_ancestry(plan_sha):
    suffix=['--kill-after=10s','900s']
    tail=['-m',COT_MODULE,'--cot-publication-correction','--plan-sha256',plan_sha]
    pid=os.getppid()
    for _ in range(64):
        cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
        argv=[os.fsdecode(part)for part in cmd if part]
        if argv and Path(argv[0]).name=='timeout':
            need(len(argv)==9 and argv[1:3]==suffix and argv[4:]==tail
                and argv[3]in ('.venv/bin/python',str(REPO/'.venv/bin/python')),'COT_EXACT_OUTER_TIMEOUT')
            return {'argv':argv,'pid':pid,'outer_seconds':900,'kill_after_seconds':10}
        status=Path('/proc/'+str(pid)+'/status').read_text().splitlines()
        parents=[line.split(':',1)[1].strip()for line in status if line.startswith('PPid:')]
        need(len(parents)==1 and int(parents[0])>0,'COT_OUTER_TIMEOUT_MISSING');pid=int(parents[0])
    need(False,'COT_OUTER_TIMEOUT_MISSING')


def cot_bookend(plan,plan_ref,policy_ref):
    return cot_source_clean(plan)and cot_json_ref(POLICY)==policy_ref and cot_json_ref(COT_PLAN)==plan_ref


def cot_run(plan_sha):
    began=time.monotonic();no_torch();firewall=install_native_m5_direction_io_firewall()
    cap=owner.require_capped_cpu_research_producer_execution();timeout=cot_timeout_ancestry(plan_sha)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'COT_TIMER_ALREADY_ACTIVE')
    plan,plan_ref,policy_ref=cot_inspect(plan_sha,admitted=True);COT_ROOT.mkdir(exist_ok=False)
    claim=cot_publish('CLAIM.json',{'stage':COT_STAGE,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'capped_execution':cap,
        'outer_timeout':timeout,'training_authorized':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_TA_COT_INTERNAL_870S')
    try:
        remaining=870-(time.monotonic()-began);need(remaining>0,'COT_TIME_BEFORE_DECODE')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=cot_produce(plan);cot_checked_inputs(plan,payloads=True)
        need(report['actual_counts']==COT_COUNTS and cot_bookend(plan,plan_ref,policy_ref)
            and firewall['denied_attempts']==0,'COT_SOURCE_POLICY_COUNTS_FIREWALL_BOOKEND')
        result=cot_publish('RESULT.json',{'stage':COT_STAGE,'status':'COMPLETE_NATIVE_M5_TA_COT_CLOCK_CORRECTION',
            'plan':plan_ref,'claim':claim,'report':report,'source_unchanged':True,'actual_counts':report['actual_counts'],
            'clean_process_io_firewall':firewall,'training_authorized':False,'test_data_used':False,
            'elapsed_seconds':time.monotonic()-began})
        terminal=cot_publish('TERMINAL.json',{'stage':COT_STAGE,'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':True,
            'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        try:unchanged=cot_bookend(plan,plan_ref,policy_ref)
        except BaseException:unchanged=False
        cot_publish('TERMINAL.json',{'stage':COT_STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,'source_unchanged':unchanged,
            'error':type(error).__name__+':'+str(error)[:2000],'remaining_invocations':0,'relaunch_allowed':False,
            'training_authorized':False,'test_data_used':False})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def cot_main(argv):
    parser=argparse.ArgumentParser();parser.add_argument('--cot-publication-correction',action='store_true',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args(argv)
    need(argv==['--cot-publication-correction','--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else []),'COT_EXACT_CLI')
    no_torch();install_native_m5_direction_io_firewall()
    if args.preflight:
        _,plan_ref,policy_ref=cot_inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_TA_COT_METADATA_ONLY','plan':plan_ref,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'cached_values_decoded':0,'Torch_imported':False,
            'normalization_fits':0,'model_constructions':0,'optimizer_steps':0}))
    else:print(json.dumps(cot_run(args.plan_sha256),sort_keys=True))


# Separate statistics qualification; the corrected cache is the sole data authority.
COT_NORM_STAGE='native_m5_ta_cot_clock_normalization_20261121_001'
COT_NORM_PARENT=COT_PARENT
COT_NORM_GATE='cot_normalization_001'
COT_NORM_PLAN=REPO/'configs/research/NATIVE_M5_TA_COT_CLOCK_NORMALIZATION_20261121_001.json'
COT_NORM_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_M5_TA_COT_CLOCK_NORMALIZATION_20261121_001')
COT_NORM_EXECUTION=dict(COT_EXECUTION)
COT_NORM_COUNTS={**COUNTS,'normalization_fits':1,'prior_fits':0,'snapshot_loads':1,
    'cached_arrays_decoded':27,'TA_rebuilds':0,'macro_materializations':0,
    'normalization_publications':1,'input_binding_publications':1,
    'normalization_readbacks':1,'input_binding_readbacks':1}
COT_NORM_INPUTS={
    'cache_result':{'path':str(COT_ROOT/'RESULT.json'),'sha256':'6dd7df0fd2f0f2b736ba8034697e1c6a8f5e2e9c20138d76f3154568dffec206'},
    'cache_terminal':{'path':str(COT_ROOT/'TERMINAL.json'),'sha256':'8eaef41cb506f40b20e4c1d0cfd6c76d6469347c211452ab900f42ff8765d50c'},
    'old_normalization':{'path':'/home/andre2/GX1_RUNS/NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001/NORMALIZATION.json',
        'sha256':'157b9604970434590e2745bf62c182d53c51e39c7e6c37c6b070b8631c09e993'}}
COT_NORM_PRODUCER_COMMIT='6fc5453ab69f753b57add459d087765e6fc7b1c5'
COT_NORM_CHANGED_SOURCE_OWNERS=('gx1/audit/build_native_m5_ta_feature_cache_v1.py',
    'tests/test_build_native_m5_ta_feature_cache_v1.py','gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh')
COT_NORM_VECTOR_STATS=('center','scale','binary_mask','categorical_mask','scale_source',
    'train_transformed_min','train_transformed_max')


def cot_norm_parent_source_bridge(parent):
    need(set(parent['source_owners'])==set(COT_SOURCE_PATHS),'COT_NORM_PARENT_SOURCE_SET')
    preserved={}
    for name,value in parent['source_owners'].items():
        if name not in COT_NORM_CHANGED_SOURCE_OWNERS:
            need(sha(REPO/name)==value,'COT_NORM_NUMERIC_SOURCE_CHANGED:'+name);preserved[name]=value
    path='gx1/audit/build_native_m5_ta_feature_cache_v1.py'
    old_raw=subprocess.check_output(['git','show',COT_NORM_PRODUCER_COMMIT+':'+path],cwd=REPO)
    need(hashlib.sha256(old_raw).hexdigest()==parent['source_owners'][path],'COT_NORM_PARENT_DRIVER_GIT')
    old=ast.parse(old_raw);current=ast.parse((REPO/path).read_text())
    old_functions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
    functions={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
    for name,node in old_functions.items():
        need(name in functions,'COT_NORM_OLD_FUNCTION_MISSING')
        actual=functions[name]
        if name=='main':
            expected=ast.parse("if '--cot-normalization' in sys.argv[1:]:\n    return cot_norm_main(sys.argv[1:])").body[0]
            need(ast.dump(actual.body[0])==ast.dump(expected),'COT_NORM_EXACT_DISPATCH')
            actual=ast.FunctionDef(name=actual.name,args=actual.args,body=actual.body[1:],
                decorator_list=actual.decorator_list,returns=actual.returns,type_comment=actual.type_comment)
        need(ast.dump(node,include_attributes=False)==ast.dump(actual,include_attributes=False),
            'COT_NORM_OLD_FUNCTION_CHANGED:'+name)
    return {'producer_commit':COT_NORM_PRODUCER_COMMIT,'parent_driver_sha256':hashlib.sha256(old_raw).hexdigest(),
        'preserved_parent_source_hashes':preserved,'explicit_changed_control_owners':list(COT_NORM_CHANGED_SOURCE_OWNERS),
        'old_functions_AST_preserved_except_exact_norm_dispatch':len(old_functions)}


def cot_norm_metadata():
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    r=cot_read(COT_NORM_INPUTS['cache_result']);t=cot_read(COT_NORM_INPUTS['cache_terminal'])
    need(r['stage']==t['stage']==COT_STAGE and r['status']=='COMPLETE_NATIVE_M5_TA_COT_CLOCK_CORRECTION'
        and t['status']=='COMPLETE'and t['exit_code']==0 and t['result']==COT_NORM_INPUTS['cache_result']
        and r['source_unchanged']is t['source_unchanged']is True
        and r['test_data_used']is t['test_data_used']is False
        and r['training_authorized']is t['training_authorized']is False
        and r['plan']==t['plan']and r['claim']==t['claim'],'COT_NORM_COMPLETE_PARENT')
    parent=cot_read(r['plan']);claim=cot_read(r['claim']);report=r['report']
    need(parent['stage']==claim['stage']==COT_STAGE and claim['plan']==r['plan']
        and r['actual_counts']==report['actual_counts']==COT_COUNTS
        and report['COT_clock_witness']['matched_rows']==21,'COT_NORM_PARENT_PROOF')
    snapshot=cot_read(report['snapshot']);contract=features.native_m5_feature_provider_contract()
    metadata=snapshot['surfaces_metadata'];axes=metadata['axes']
    need(snapshot['schema_version']=='gx1_native_m5_bounded_feature_snapshot_v1'and len(snapshot['arrays'])==27
        and snapshot['archive']==report['archive']
        and snapshot['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256']is True
        and metadata['provider_contract']==report['input_provider_contract']==contract
        and metadata['macro_binding']==report['macro_binding']and axes==report['source_axes']
        and axes['contract_sha256']==features._sha({k:v for k,v in axes.items()if k!='contract_sha256'})
        and axes['issue_start_ns']==START_NS and axes['issue_end_ns']==END_NS
        and report['TRAIN_rows']==354550 and report['development_rows']==34934,
        'COT_NORM_CORRECTED_SNAPSHOT')
    old=norm.require_native_m5_input_normalization_contract(cot_read(COT_NORM_INPUTS['old_normalization']))
    need(old['native_m5_manifest']==metadata['source_binding']['native_m5_manifest']
        and old['source_axes_sha256']!=axes['contract_sha256']
        and old['fit_population']['issue_rows']==354550 and old['fit_population']['cutoff_ns']==1767225600000000000
        and old['fit_population']['selected_issue_ns_sha256']==report['TRAIN_issue_ns_sha256'],
        'COT_NORM_REFERENCE_POPULATION_ONLY')
    dependencies={}
    for binding in [*COT_NORM_INPUTS.values(),r['plan'],r['claim'],report['snapshot'],
            metadata['source_binding']['native_m5_manifest'],
            parent['metadata']['current_source_spec'],parent['metadata']['current_clock']]:
        cot_read(binding);dependencies[binding['path']]=binding
    return {'dependencies':dependencies,'snapshot':report['snapshot'],'archive':cot_binding(report['archive']),
        'source_axes':axes,'feature_contract':contract,'old_normalization':COT_NORM_INPUTS['old_normalization'],
        'fit_population':old['fit_population'],'TRAIN_issue_ns_sha256':report['TRAIN_issue_ns_sha256'],
        'parent_source_bridge':cot_norm_parent_source_bridge(parent)}


def cot_norm_build_plan(*,source_base_commit=None):
    metadata=cot_norm_metadata()
    head=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    return {'schema_version':'gx1_native_m5_ta_cot_clock_normalization_plan_v1','stage':COT_NORM_STAGE,
        'status':'PREPARED_NOT_ADMITTED','output_root':str(COT_NORM_ROOT),'source_base_commit':head,
        'source_owners':{name:sha(REPO/name)for name in COT_SOURCE_PATHS},'inputs':COT_NORM_INPUTS,
        'metadata':metadata,'execution':COT_NORM_EXECUTION,'limits':COT_NORM_COUNTS,
        'training_authorized':False,'test_data_used':False,'GC_used':False,'M1_used':False}


def cot_norm_checked_inputs(plan,*,payloads):
    for binding in plan['metadata']['dependencies'].values():cot_read(binding)
    if payloads:
        binding=plan['metadata']['archive'];need(ref(binding['path'])==binding,'COT_NORM_ARCHIVE_BOOKEND')
    no_torch()


def cot_norm_inspect(plan_sha,*,admitted):
    need(Path.cwd().resolve()==REPO and not COT_NORM_ROOT.exists()and not COT_NORM_ROOT.is_symlink(),'COT_NORM_CWD_OUTPUT')
    pr=cot_binding({'path':str(COT_NORM_PLAN),'sha256':plan_sha},json_only=True);plan=cot_read(pr)
    need(type(plan.get('source_owners'))is dict and set(plan['source_owners'])==set(COT_SOURCE_PATHS),'COT_NORM_SOURCE_SET')
    need(plan==cot_norm_build_plan(source_base_commit=plan['source_base_commit'])and cot_source_clean(plan),'COT_NORM_PLAN_SOURCE')
    policy_ref=cot_json_ref(POLICY);policy=cot_read(policy_ref);parent=policy.get(COT_NORM_PARENT,{})
    gate=parent.get(COT_NORM_GATE,{})
    need(all(policy.get(k)is False for k in owner.data.CLOSED_FLAGS)
        and parent.get('goal_status')=='ACTIVE'and parent.get('large_training_qualified')is False
        and gate.get('plan')==pr and gate.get('output_root')==str(COT_NORM_ROOT)
        and gate.get('execution')==COT_NORM_EXECUTION and gate.get('limits')==COT_NORM_COUNTS
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed')is False and gate.get('training_authorized')is False
        and gate.get('test_authorized')is False,'COT_NORM_CLOSED_POLICY')
    expected=('ADMITTED_ONE_USE',True,1)if admitted else('PREPARED_NOT_ADMITTED',False,0)
    need((gate.get('status'),gate.get('execution_authorized'),gate.get('remaining_invocations'))==expected,'COT_NORM_ONE_USE')
    cot_norm_checked_inputs(plan,payloads=False);return plan,pr,policy_ref


def cot_norm_stats_comparison(old,new):
    need(set(old)==set(new)and old['fit_population']==new['fit_population'],'COT_NORM_EXACT_POPULATION')
    for key in old:
        if key not in ('contract_sha256','source_axes_sha256','feature_provider_contract_sha256','surfaces'):
            need(old[key]==new[key],'COT_NORM_UNCHANGED_CONTRACT:'+key)
    compared={}
    need(set(old['surfaces'])==set(new['surfaces']),'COT_NORM_SURFACE_SET')
    for name,previous in old['surfaces'].items():
        current=new['surfaces'][name]
        if name not in ('signal','ctx_cont'):
            need(previous==current,'COT_NORM_UNCHANGED_MTF:'+name);compared[name]=previous['field_count'];continue
        fields=previous['field_names'];allowed={'ctx_cont.'+f if name=='signal'else f for f in COT_FIELDS}
        need(set(previous)==set(current)and fields==current['field_names'],'COT_NORM_STATS_KEYS')
        for key,value in previous.items():
            if key=='stats_sha256':continue
            if key in COT_NORM_VECTOR_STATS:
                need(len(value)==len(current[key])==len(fields)and all(value[i]==current[key][i]
                    for i,field in enumerate(fields)if field not in allowed),'COT_NORM_NONCOT_STATS:'+name+':'+key)
            else:need(value==current[key],'COT_NORM_UNCHANGED_STATS:'+name+':'+key)
        compared[name]=len(fields)-len(allowed)
    return {'all_nonCOT_perfield_statistics_exact':True,'all_five_MTF_surface_contracts_exact':True,
        'ctx_cat_temporal_aliases_population_exact':True,'unchanged_field_counts':compared}


def cot_norm_local_union(rows,length):
    """Independent interval merge, avoiding a second call to the fitter's union helper."""
    need(rows.dtype==np.int64 and rows.ndim==1 and len(rows)>0 and np.all(np.diff(rows)>0)
        and type(length)is int and length>0 and rows[0]>=length-1,'COT_NORM_LOCAL_UNION')
    start=int(rows[0])-length+1;end=int(rows[0]);parts=[]
    for tail in rows[1:]:
        left=int(tail)-length+1
        if left>end+1:parts.append(np.arange(start,end+1,dtype=np.int64));start=left
        end=int(tail)
    parts.append(np.arange(start,end+1,dtype=np.int64))
    return np.concatenate(parts)


def cot_norm_four_stats_witness(provider,rows,normalization):
    fields=provider.profile['physical_field_names'];local=cot_norm_local_union(rows,features.MODEL_NATIVE_SEQ_LEN)
    local_positions=provider._feature_positions(local);positions=provider._feature_positions(rows);reports={}
    need(len(local)==normalization['fit_population']['local_rows'],'COT_NORM_LOCAL_POPULATION')
    for field in COT_FIELDS:
        si=fields['signal'].index('ctx_cont.'+field);ci=fields['ctx_cont'].index(field)
        raw=provider.surfaces['signal'][local_positions,si].astype(np.float64)
        context=provider.surfaces['ctx_cont'][positions,ci].astype(np.float64)
        need(np.isfinite(raw).all()and np.isfinite(context).all(),'COT_NORM_WITNESS_FINITE')
        q25,median,q75=np.quantile(raw,[.25,.5,.75]);center=np.float32(median);scale=np.float32(q75-q25);source='raw_iqr'
        if not np.isfinite(scale)or scale<=np.float32(0):
            positive=np.abs(raw-float(center));positive=positive[positive>0]
            need(len(positive)>=2,'COT_NORM_WITNESS_UNSCALEABLE')
            scale=np.float32(np.median(positive));source='median_positive_abs_deviation'
        need(np.isfinite(scale)and scale>np.float32(0),'COT_NORM_WITNESS_UNSCALEABLE')
        supports={}
        for name,column,index in (('signal',raw,si),('ctx_cont',context,ci)):
            transformed=np.arcsinh((column-float(center))/float(scale));surface=normalization['surfaces'][name]
            expected={'center':float(center),'scale':float(scale),'scale_source':source,'binary_mask':0,'categorical_mask':0,
                'train_transformed_min':float(transformed.min()),'train_transformed_max':float(transformed.max())}
            need(all(surface[key][index]==value for key,value in expected.items()),'COT_NORM_FOUR_STATS:'+name+':'+field)
            supports[name]={'population_rows':len(column),'raw_F64_sha256':digest(column),**expected}
        reports[field]=supports
    return {'independent_columns':4,'formula':'F64 quantile(.25,.5,.75); F32 median/IQR; canonical positive-deviation fallback; F64 asinh support',
        'local_union_rows':len(local),'local_union_source_rows_sha256':digest(local),'per_field':reports}


def cot_norm_publish(name,value):
    need(name in ('CLAIM.json','NORMALIZATION.json','INPUT_BINDING.json','RESULT.json','TERMINAL.json'),'COT_NORM_OUTPUT_NAME')
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();path=COT_NORM_ROOT/name;_publish_bytes(path,raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def cot_norm_produce(plan):
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    meta=plan['metadata'];counts={k:0 for k in COT_NORM_COUNTS};cot_norm_checked_inputs(plan,payloads=True)
    provider=load_snapshot(meta['snapshot']);counts['snapshot_loads']=1;counts['cached_arrays_decoded']=27
    old=norm.require_native_m5_input_normalization_contract(cot_read(meta['old_normalization']))
    need(provider.surfaces['axes']==meta['source_axes']and provider.surfaces['provider_contract']==meta['feature_contract'],
        'COT_NORM_CURRENT_PROVIDER')
    all_rows=provider.surfaces['retained_issue_rows'];issues=provider.issue_ns[all_rows]
    rows=all_rows[(issues>=START_NS)&(issues<1767225600000000000)]
    need(len(rows)==354550 and provider.input_ready[all_rows].all()
        and digest(provider.issue_ns[rows])==meta['TRAIN_issue_ns_sha256']
        and digest(rows)==old['fit_population']['selected_source_rows_sha256'],'COT_NORM_WHOLE_TRAIN_INPUT_AXIS')
    normalization=features.fit_native_m5_feature_normalization(provider,fit_source_rows=rows,cutoff_ns=1767225600000000000)
    counts['normalization_fits']+=1
    norm.require_native_m5_input_normalization_contract(normalization)
    comparison=cot_norm_stats_comparison(old,normalization)
    witness=cot_norm_four_stats_witness(provider,rows,normalization)
    binding=provider.input_binding(normalization)
    need(binding['source_axes_sha256']==meta['source_axes']['contract_sha256']
        and binding['feature_provider_contract_sha256']==meta['feature_contract']['contract_sha256']
        and normalization['fit_population']==meta['fit_population'],'COT_NORM_NEW_INPUT_BINDING')
    norm_ref=cot_norm_publish('NORMALIZATION.json',normalization);counts['normalization_publications']=1
    binding_ref=cot_norm_publish('INPUT_BINDING.json',binding);counts['input_binding_publications']=1
    need(cot_json_ref(norm_ref['path'])==norm_ref,'COT_NORM_DISK_HASH')
    saved=norm.require_native_m5_input_normalization_contract(cot_read(norm_ref));counts['normalization_readbacks']=1
    need(saved==normalization and cot_json_ref(binding_ref['path'])==binding_ref,'COT_NORM_DISK_READBACK')
    saved_binding=cot_read(binding_ref);counts['input_binding_readbacks']=1
    need(saved_binding==binding==provider.input_binding(saved)and counts==COT_NORM_COUNTS,'COT_NORM_BINDING_READBACK_COUNTS')
    no_torch()
    return {'normalization':norm_ref,'input_binding':binding_ref,'snapshot':meta['snapshot'],
        'archive':meta['archive'],'fit_population':saved['fit_population'],'TRAIN_issue_ns_sha256':meta['TRAIN_issue_ns_sha256'],
        'source_axes_sha256':provider.source_axes_sha256,'statistics_comparison':comparison,
        'COT_statistics_witness':witness,'actual_counts':counts,'Torch_imported':False,'test_data_used':False,
        'initialbaseline_qualified':False,'prior_refit_performed':False,'old_normalization_reference_only':True}


def cot_norm_timeout_ancestry(plan_sha):
    tail=['--kill-after=10s','900s'];command=['-m',COT_MODULE,'--cot-normalization','--plan-sha256',plan_sha]
    pid=os.getppid()
    for _ in range(64):
        argv=[os.fsdecode(v)for v in Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')if v]
        if argv and Path(argv[0]).name=='timeout':
            need(len(argv)==9 and argv[1:3]==tail and argv[4:]==command
                and argv[3]in ('.venv/bin/python',str(REPO/'.venv/bin/python')),'COT_NORM_EXACT_OUTER_TIMEOUT')
            return {'argv':argv,'pid':pid,'outer_seconds':900,'kill_after_seconds':10}
        parents=[line.split(':',1)[1].strip()for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines()if line.startswith('PPid:')]
        need(len(parents)==1 and int(parents[0])>0,'COT_NORM_OUTER_TIMEOUT_MISSING');pid=int(parents[0])
    need(False,'COT_NORM_OUTER_TIMEOUT_MISSING')


def cot_norm_bookend(plan,pr,policy_ref):
    return cot_source_clean(plan)and cot_json_ref(POLICY)==policy_ref and cot_json_ref(COT_NORM_PLAN)==pr


def cot_norm_run(plan_sha):
    began=time.monotonic();no_torch();firewall=install_native_m5_direction_io_firewall()
    cap=owner.require_capped_cpu_research_producer_execution();timeout=cot_norm_timeout_ancestry(plan_sha)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'COT_NORM_TIMER_ALREADY_ACTIVE')
    plan,pr,policy_ref=cot_norm_inspect(plan_sha,admitted=True);COT_NORM_ROOT.mkdir(exist_ok=False)
    claim=cot_norm_publish('CLAIM.json',{'stage':COT_NORM_STAGE,'status':'CLAIMED_ONE_USE','plan':pr,
        'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'capped_execution':cap,
        'outer_timeout':timeout,'training_authorized':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_TA_COT_NORM_INTERNAL_870S')
    try:
        remaining=870-(time.monotonic()-began);need(remaining>0,'COT_NORM_TIME_BEFORE_DECODE')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=cot_norm_produce(plan);cot_norm_checked_inputs(plan,payloads=True)
        need(report['actual_counts']==COT_NORM_COUNTS and cot_norm_bookend(plan,pr,policy_ref)
            and firewall['denied_attempts']==0,'COT_NORM_SOURCE_POLICY_COUNTS_FIREWALL_BOOKEND')
        result=cot_norm_publish('RESULT.json',{'stage':COT_NORM_STAGE,'status':'COMPLETE_NATIVE_M5_TA_COT_CLOCK_NORMALIZATION',
            'plan':pr,'claim':claim,'report':report,'source_unchanged':True,'actual_counts':report['actual_counts'],
            'clean_process_io_firewall':firewall,'training_authorized':False,'test_data_used':False,
            'elapsed_seconds':time.monotonic()-began})
        terminal=cot_norm_publish('TERMINAL.json',{'stage':COT_NORM_STAGE,'status':'COMPLETE','exit_code':0,
            'plan':pr,'claim':claim,'result':result,'source_unchanged':True,'remaining_invocations':0,
            'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        try:unchanged=cot_norm_bookend(plan,pr,policy_ref)
        except BaseException:unchanged=False
        cot_norm_publish('TERMINAL.json',{'stage':COT_NORM_STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':pr,'claim':claim,'result':result,'source_unchanged':unchanged,'remaining_invocations':0,
            'relaunch_allowed':False,'training_authorized':False,'test_data_used':False,
            'error':type(error).__name__+':'+str(error)[:2000]})
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def cot_norm_main(argv):
    parser=argparse.ArgumentParser();parser.add_argument('--cot-normalization',action='store_true',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args(argv)
    need(argv==['--cot-normalization','--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else []),'COT_NORM_EXACT_CLI')
    no_torch();install_native_m5_direction_io_firewall()
    if args.preflight:
        _,pr,policy_ref=cot_norm_inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_TA_COT_NORMALIZATION_METADATA_ONLY','plan':pr,'policy':policy_ref,
            'claim_created':False,'raw_values_decoded':0,'cached_values_decoded':0,'Torch_imported':False,
            'normalization_fits':0,'model_constructions':0,'optimizer_steps':0}))
    else:print(json.dumps(cot_norm_run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
