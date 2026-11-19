"""Fresh native GC source bindings and mask-preserving joint direction inputs.

Original raw components and numeric packet provenance are admitted by the
assembly owner. Model validators are pure: no legacy registry, source table,
normalization plan or artifact is discovered through a field name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from gx1.features import gc_native_m5_features_v1 as gc
from gx1.contracts import market_state_native_m5_feature_inputs_v1 as ta
from gx1.contracts.entry_model_native_input_normalization_v1 import NATIVE_M5_TRAIN_FIT_CUTOFF_NS

SOURCE_SCHEMA = 'gx1_gc_native_m5_feature_source_binding_v1'
INPUT_SCHEMA = 'gx1_native_m5_full_gc_bar_direction_inputs_v1'
PACKET_SCHEMA = 'gx1_gc_native_m5_numeric_packet_v1'
PROOF_SCHEMA = 'gx1_gc_native_m5_assembly_proof_v1'
CLOCK = 'native_m5_and_closed_higher_no_m1_v1'
TIMEFRAMES = gc.TIMEFRAMES


def _need(value, reason):
    if not bool(value):raise RuntimeError('NATIVE_M5_GC_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _seal(value):
    return {**value,'contract_sha256':_sha(value)}


def _digest(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _sealed(value, reason):
    _need(isinstance(value,dict) and value.get('contract_sha256')==_sha(
        {k:v for k,v in value.items() if k!='contract_sha256'}),reason)


def _ref(value):
    _need(isinstance(value,dict) and set(value)=={'path','sha256'} and _digest(value['sha256']), 'REF')
    p=Path(value['path'])
    _need(p.is_absolute() and str(p)==value['path'] and '..' not in p.parts
        and 'NATIVE_GC_CONTEXT_FIELDS' not in str(p) and 'XAU_M1' not in str(p)
        and re.search(r'(?<![A-Z0-9])M1(?![A-Z0-9])',str(p).upper()) is None,'REF_PATH')
    return dict(value)


def _file_ref(value):
    ref=_ref(value);p=Path(ref['path'])
    _need(p.is_file() and not p.is_symlink() and p.resolve(strict=True)==p,'REF_FILE')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    _need(h.hexdigest()==ref['sha256'],'REF_HASH')
    return p


def _json_ref(value):
    return json.loads(_file_ref(value).read_text())


def _packet_layout(packet):
    _need(packet.get('schema_version')==gc.SCHEMA_VERSION and packet.get('input_interval_ns')==gc.INTERVAL_NS,
          'PACKET_IDENTITY')
    gc._clock(packet['source_clock'])
    arrays={};layout={};axes={}
    for group in ('timeframes','m5_book_features'):
        tables=packet[group] if group=='timeframes' else {'M5':packet[group]}
        _need(set(tables)==(set(TIMEFRAMES) if group=='timeframes' else {'M5'}),'PACKET_TF')
        for tf,table in tables.items():
            starts=np.asarray(table['bar_start_ns']);available=np.asarray(table['available_ns'])
            _need(starts.dtype==available.dtype==np.int64 and starts.ndim==available.ndim==1
                and starts.shape==available.shape and np.all(np.diff(starts)>0)
                and np.all(available<gc.source.timing.WINDOW_END_NS),'PACKET_CLOCK')
            axes[group+'/'+tf]={'rows':len(starts),'bar_start_ns_sha256':_array_sha(starts),
                               'available_ns_sha256':_array_sha(available)}
            for name,raw in table.items():
                array=np.asarray(raw);path=[group,tf,name] if group=='timeframes' else [group,name]
                _need(array.dtype.kind in 'biuf' and array.shape==starts.shape,'PACKET_ARRAY')
                key='__'.join(path);arrays[key]=array
                layout[key]={'path':path,'dtype':array.dtype.str,'shape':list(array.shape),'sha256':_array_sha(array)}
    return arrays,layout,axes


def require_native_m5_gc_source_binding(value):
    """Validate the sealed new-native identity before any nested artifact IO."""
    _need(isinstance(value,dict) and value.get('schema_version')==SOURCE_SCHEMA,'SOURCE_SCHEMA')
    _sealed(value,'SOURCE_HASH')
    keys={'schema_version','source_clock_contract','native_field_catalog_sha256','unit_policy_sha256',
        'source_clock','raw_source_identity_sha256','native_xau_m5_source_binding_sha256','native_xau_m5_manifest',
        'assembly_proof','raw_components_result','packet_artifacts','packet_array_contract_sha256',
        'packet_axes','axes_sha256','contract_sha256'}
    catalog=gc.native_m5_field_catalog()
    _need(set(value)==keys and value['source_clock_contract']==CLOCK
        and value['native_field_catalog_sha256']==catalog['contract_sha256']
        and value['unit_policy_sha256']==catalog['unit_policy_sha256'],'SOURCE_IDENTITY')
    gc._clock(value['source_clock'])
    for key in ('raw_source_identity_sha256','native_xau_m5_source_binding_sha256',
                'packet_array_contract_sha256','axes_sha256'):
        _need(_digest(value[key]),'SOURCE_DIGEST')
    for key in ('assembly_proof','raw_components_result','native_xau_m5_manifest'):_ref(value[key])
    _need(set(value['packet_artifacts'])=={'json','npz'},'PACKET_REFS')
    for ref in value['packet_artifacts'].values():_ref(ref)
    _need(set(value['packet_axes'])=={'timeframes/'+tf for tf in TIMEFRAMES}|{'m5_book_features/M5'}
        and value['axes_sha256']==_sha(value['packet_axes']),'SOURCE_AXES')
    for axis in value['packet_axes'].values():
        _need(set(axis)=={'rows','bar_start_ns_sha256','available_ns_sha256'} and type(axis['rows']) is int
            and axis['rows']>=0 and _digest(axis['bar_start_ns_sha256'])
            and _digest(axis['available_ns_sha256']),'SOURCE_AXIS')
    return json.loads(json.dumps(value))


def _raw_component_lineage(proof,raw_source_binding):
    result_ref=proof['raw_components_result']
    _need(Path(_ref(result_ref)['path']).name=='RESULT.json'
        and Path(result_ref['path']).parent.name.startswith('GC_NATIVE_M5_RAW_COMPONENTS_'),'RAW_RESULT_NAMESPACE')
    result=_json_ref(result_ref)
    _need(result.get('status')=='COMPLETE_RAW_COMPONENTS_ONLY' and result.get('source_unchanged') is True
        and result.get('all_raw_payload_hash_bookends_unchanged') is True
        and result.get('native_input_interval_ns')==gc.INTERVAL_NS and result.get('test_data_used') is False
        and result.get('model_forwards')==result.get('optimizer_steps')==0
        and result.get('files')==proof['raw_components'] and len(result['files'])==66,'RAW_RESULT')
    plan_ref,claim_ref=_ref(result['plan']),_ref(result['claim'])
    _need(Path(plan_ref['path']).name.startswith('GC_NATIVE_M5_RAW_COMPONENTS_')
        and Path(claim_ref['path']).name=='CLAIM.json','RAW_PLAN_CLAIM_NAMESPACE')
    plan,claim=_json_ref(plan_ref),_json_ref(claim_ref)
    _need(plan.get('schema_version')=='gx1_gc_native_m5_raw_components_plan_v1'
        and plan.get('source_binding')==raw_source_binding and plan.get('test_data_used') is False
        and claim.get('plan')==plan_ref and claim.get('status')=='CLAIMED_ONE_USE'
        and claim.get('test_data_used') is False,'RAW_PLAN_CLAIM')
    _need(proof['raw_components_plan']==plan_ref and proof['raw_components_claim']==claim_ref,'RAW_PROOF_PLAN_CLAIM')
    terminal=_json_ref(proof['raw_components_terminal'])
    _need(terminal.get('status')=='COMPLETE' and terminal.get('exit_code')==0
        and terminal.get('result')==result_ref and terminal.get('plan')==plan_ref
        and terminal.get('claim')==claim_ref and terminal.get('test_data_used') is False,'RAW_TERMINAL')
    seen=set()
    for ref in result['files']:
        item=_json_ref(ref);ordinal=item.get('ordinal')
        _need(item.get('schema_version')=='gx1_gc_native_m5_raw_file_components_v1'
            and item.get('status')=='COMPLETE_RAW_COMPONENTS_NOT_FEATURE_ADMISSION'
            and type(ordinal) is int and 0<=ordinal<66 and ordinal not in seen
            and item.get('source')==raw_source_binding['raw_files'][ordinal]
            and item.get('source_identity_sha256')==raw_source_binding['source_identity_sha256']
            and item.get('producer_plan')==plan_ref and item.get('producer_claim')==claim_ref
            and item.get('raw_hash_bookends_unchanged') is True
            and item.get('ladder_and_trade_size_integer_conservation') is True,'RAW_COMPONENT_LINEAGE')
        begin,end,interval=gc.source.reduction_clock_v1(item['source_clock'])
        raw_item=raw_source_binding['raw_files'][ordinal]
        root_begin=raw_source_binding['source_clock']['start_ns']
        _need(begin==raw_item['start_ns'] and end==raw_item['end_ns']
            and interval==gc.INTERVAL_NS and item.get('global_bucket_offset')==(begin-root_begin)//interval,
            'RAW_COMPONENT_SOURCE_INTERVAL')
        seen.add(ordinal)
    _need(len(seen)==66,'RAW_COMPONENTS_COMPLETE')


def bind_native_m5_gc_feature_packet(packet,*,raw_source_binding,native_xau_m5_source_binding,
                                     assembly_proof_ref,artifact_refs):
    """Admit the numeric packet through the actual raw/assembly receipts."""
    catalog=gc.native_m5_field_catalog()
    _need(raw_source_binding.get('schema_version')=='gx1_gc_native_m5_raw_source_binding_v1'
        and raw_source_binding.get('m1_data_or_metadata_used') is False
        and raw_source_binding.get('derived_feature_parents')==[]
        and raw_source_binding.get('source_identity_sha256')==_sha(
            {k:v for k,v in raw_source_binding.items() if k!='source_identity_sha256'}),'RAW_BINDING')
    ta._sealed(native_xau_m5_source_binding,'GC_XAU_SOURCE_HASH')
    _ref(native_xau_m5_source_binding['native_m5_manifest'])
    _need(native_xau_m5_source_binding.get('schema_version')==ta.SOURCE_SCHEMA
        and native_xau_m5_source_binding.get('source_clock_contract')==CLOCK,'XAU_SOURCE')
    raw_begin,raw_end,_=gc.source.reduction_clock_v1(raw_source_binding['source_clock'])
    begin,end,_=gc.source.reduction_clock_v1(packet['source_clock'])
    _need(raw_begin<=begin<end<=raw_end
        and native_xau_m5_source_binding['source_spec']['cutoff_ns']>=end,'SOURCE_SCOPE_JOIN')
    _need(set(artifact_refs)=={'json','npz'},'ARTIFACT_REFS')
    proof=_json_ref(assembly_proof_ref)
    keys={'schema_version','status','source_identity_sha256','source_clock',
        'native_xau_m5_source_binding_sha256','raw_components','raw_components_result',
        'raw_components_plan','raw_components_claim','raw_components_terminal',
        'm1_data_or_metadata_used','raw_replay','packet_artifacts','catalog_contract_sha256',
        'unit_policy_sha256','source_unchanged','test_data_used'}
    _need(set(proof)==keys and proof['schema_version']==PROOF_SCHEMA and proof['status']=='COMPLETE'
        and proof['source_identity_sha256']==raw_source_binding['source_identity_sha256']
        and proof['source_clock']==packet['source_clock']
        and proof['native_xau_m5_source_binding_sha256']==native_xau_m5_source_binding['contract_sha256']
        and proof['m1_data_or_metadata_used'] is False and proof['raw_replay'] is False
        and proof['source_unchanged'] is True and proof['test_data_used'] is False
        and proof['catalog_contract_sha256']==catalog['contract_sha256']
        and proof['unit_policy_sha256']==catalog['unit_policy_sha256']
        and proof['packet_artifacts']==artifact_refs,'ASSEMBLY_PROOF')
    _raw_component_lineage(proof,raw_source_binding)
    gc.source_coverage_report(packet)
    arrays,layout,axes=_packet_layout(packet)
    meta=_json_ref(artifact_refs['json']);_file_ref(artifact_refs['npz'])
    expected_meta={k:v for k,v in packet.items() if k not in ('timeframes','m5_book_features')}
    _need(meta.get('schema_version')==PACKET_SCHEMA and meta.get('packet_metadata')==expected_meta
        and meta.get('array_layout')==layout
        and meta.get('nullable_timestamp_columns')==list(gc.NATIVE_NULLABLE_TIMESTAMP_COLUMNS),'PACKET_METADATA')
    return require_native_m5_gc_source_binding(_seal({'schema_version':SOURCE_SCHEMA,
        'source_clock_contract':CLOCK,'native_field_catalog_sha256':catalog['contract_sha256'],
        'unit_policy_sha256':catalog['unit_policy_sha256'],'source_clock':dict(packet['source_clock']),
        'raw_source_identity_sha256':raw_source_binding['source_identity_sha256'],
        'native_xau_m5_source_binding_sha256':native_xau_m5_source_binding['contract_sha256'],
        'native_xau_m5_manifest':_ref(native_xau_m5_source_binding['native_m5_manifest']),
        'assembly_proof':_ref(assembly_proof_ref),'raw_components_result':_ref(proof['raw_components_result']),
        'packet_artifacts':{k:_ref(v) for k,v in artifact_refs.items()},
        'packet_array_contract_sha256':_sha(layout),'packet_axes':axes,'axes_sha256':_sha(axes)}))


def load_native_m5_gc_feature_packet(source_binding):
    binding=require_native_m5_gc_source_binding(source_binding)
    meta=_json_ref(binding['packet_artifacts']['json'])
    _need(meta.get('schema_version')==PACKET_SCHEMA and _sha(meta.get('array_layout'))==binding['packet_array_contract_sha256'],
          'PACKET_LOAD_LAYOUT')
    packet=dict(meta['packet_metadata']);packet['timeframes']={tf:{} for tf in TIMEFRAMES};packet['m5_book_features']={}
    path=_file_ref(binding['packet_artifacts']['npz'])
    with np.load(path,allow_pickle=False) as arrays:
        _need(set(arrays.files)==set(meta['array_layout']),'PACKET_LOAD_KEYS')
        for key,descriptor in meta['array_layout'].items():
            array=arrays[key]
            _need(array.dtype.str==descriptor['dtype'] and list(array.shape)==descriptor['shape']
                and _array_sha(array)==descriptor['sha256'],'PACKET_LOAD_ARRAY')
            target=packet
            for component in descriptor['path'][:-1]:target=target[component]
            target[descriptor['path'][-1]]=array
    _,layout,axes=_packet_layout(packet)
    _need(_sha(layout)==binding['packet_array_contract_sha256'] and axes==binding['packet_axes']
          and packet['source_clock']==binding['source_clock'],'PACKET_LOAD_BINDING')
    gc.source_coverage_report(packet)
    return packet


def require_native_m5_gc_input_binding(value):
    _need(isinstance(value,dict) and value.get('schema_version')==INPUT_SCHEMA,'INPUT_SCHEMA')
    _sealed(value,'INPUT_HASH')
    keys={'schema_version','source_bar_seconds','timeframes','source_clock_contract',
        'feature_provider_contract_sha256','source_axes_sha256','input_normalization_sha256',
        'execution_profile','warmstart_used','native_gc_source_binding_sha256',
        'gc_layout_sha256','gc_normalization_sha256','contract_sha256'}
    from gx1.features.entry_specialist_feature_groups_v1 import NATIVE_M5_FULL_GC_ENTRY_EXECUTION_PROFILE_NAME
    _need(set(value)==keys and value['source_bar_seconds']==300 and value['timeframes']==list(TIMEFRAMES)
        and value['source_clock_contract']==CLOCK and value['warmstart_used'] is False
        and set(value['execution_profile'])=={'name','contract_sha256'}
        and value['execution_profile']['name']==NATIVE_M5_FULL_GC_ENTRY_EXECUTION_PROFILE_NAME
        and _digest(value['execution_profile']['contract_sha256']),'INPUT_IDENTITY')
    for key in ('feature_provider_contract_sha256','source_axes_sha256','input_normalization_sha256',
                'native_gc_source_binding_sha256','gc_layout_sha256','gc_normalization_sha256'):
        _need(_digest(value[key]),'INPUT_DIGEST')
    return json.loads(json.dumps(value))


def native_m5_gc_feature_provider_contract(gc_mtf_history):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_native_m5_full_gc_entry_execution_profile
    from gx1.contracts.entry_gc_mtf_history_v1 import require_gc_mtf_history
    history=require_gc_mtf_history(gc_mtf_history)
    profile=compile_native_m5_full_gc_entry_execution_profile(history)
    catalog=gc.native_m5_field_catalog()
    return _seal({'schema_version':'gx1_native_m5_full_gc_feature_provider_v1',
        'source_clock_contract':CLOCK,'source_bar_seconds':300,'timeframes':list(TIMEFRAMES),
        'ta_provider_contract_sha256':ta.native_m5_feature_provider_contract()['contract_sha256'],
        'native_field_catalog_sha256':catalog['contract_sha256'],'unit_policy_sha256':catalog['unit_policy_sha256'],
        'gc_layout_sha256':history['layout']['contract_sha256'],
        'execution_profile':{'name':profile['name'],'contract_sha256':profile['contract_sha256']},
        'field_names':{k:list(v) for k,v in profile['physical_field_names'].items()},
        'local_gc':'latest_closed_native_GC_bucket_authoritative_including_unknowns',
        'own_tf_gc':'exact_TA_global_TF_row_clock_join_with_absent_GC_observation_zero',
        'gather_keys':list(sorted(profile['feature_keys']))})


def _field_values(field,table,decision_ns,*,exact=False,include_source=False):
    decisions=np.asarray(decision_ns,dtype=np.int64)
    values=np.zeros(len(decisions),np.float32);observed=np.zeros(len(decisions),bool)
    source_out=np.full(len(decisions),-1,np.int64)
    def result():return (values,observed,source_out) if include_source else (values,observed)
    clock=np.asarray(table['available_ns'])
    _need(clock.dtype==np.int64 and clock.ndim==1 and np.all(np.diff(clock)>0),'FIELD_CLOCK')
    if not len(clock):return result()
    index=np.searchsorted(clock,decisions,side='right')-1
    rows=np.flatnonzero(index>=0)
    if exact:rows=rows[clock[index[rows]]==decisions[rows]]
    if not len(rows):return result()
    at=index[rows];raw=np.asarray(table[field['column']])[at]
    source=np.asarray(table[field['source_clock_column']])[at]
    known=np.isfinite(raw)&np.isfinite(source)&(source>=0)
    if field['validity_column']:
        gate=np.asarray(table[field['validity_column']])[at]
        _need(np.isin(gate,(0,1)).all(),'FIELD_GATE')
        known&=gate.astype(bool)
    _need(not np.any(known&(source>clock[at])) and not np.any(known&(source>decisions[rows])), 'FIELD_FUTURE')
    transform=field['transform']
    if transform=='decision_age_minutes':raw=(decisions[rows]-source)/gc.MINUTE_NS
    elif transform=='native_m5_trailing_support_gap_minutes':
        _need(not np.any(known&((raw<0)|(raw+gc.INTERVAL_NS>clock[at]))),'FIELD_SUPPORT_CLOCK')
        raw=(decisions[rows]-raw-gc.INTERVAL_NS)/gc.MINUTE_NS
    else:_need(transform is None,'FIELD_TRANSFORM')
    _need(not np.any(known&~np.isfinite(raw)),'FIELD_NONFINITE')
    if field['binary_value']:_need(np.isin(raw[known],(0,1)).all(),'FIELD_BINARY')
    _need(np.all(source[known]==source[known].astype(np.int64)),'FIELD_SOURCE_INTEGER')
    values[rows[known]]=np.asarray(raw[known],np.float32);observed[rows[known]]=True
    source_out[rows[known]]=source[known].astype(np.int64)
    return result()


def _table(packet,field):
    return (packet['m5_book_features'] if field['packet_group']=='m5_book_features'
            else packet['timeframes'][field['timeframe']])


class NativeM5GCFeatureProvider:
    """Append every native GC value/mask without removing a TA decision row."""
    def __init__(self,ta_provider,packet,source_binding):
        _need(isinstance(ta_provider,ta.NativeM5FeatureProvider),'TA_PROVIDER')
        self.ta=ta_provider;self.source_binding=require_native_m5_gc_source_binding(source_binding)
        _need(self.source_binding['native_xau_m5_manifest']==self.ta.surfaces['source_binding']['native_m5_manifest'],
              'XAU_MANIFEST_JOIN')
        _,layout,axes=_packet_layout(packet)
        _need(packet['source_clock']==self.source_binding['source_clock']
            and _sha(layout)==self.source_binding['packet_array_contract_sha256']
            and axes==self.source_binding['packet_axes'],'NUMERIC_SOURCE_BINDING')
        gc.source_coverage_report(packet)
        self.catalog=gc.native_m5_field_catalog()
        self.fields=self.catalog['fields']
        self.base_fields=[f for f in self.fields if f['availability_field']==f['name']+'_observed']
        self.gc_context_names=tuple(name for f in self.base_fields for name in (f['name'],f['availability_field']))
        self.issue_ns=self.ta.issue_ns;self.source_row=self.ta.source_row
        self.retained_source_rows=self.ta.retained_source_rows
        self.input_ready=self.ta.input_ready
        # Only the TA-consumed local prefix/window union is materialized.
        decisions=self.issue_ns[self.retained_source_rows]
        context=np.zeros((len(decisions),len(self.gc_context_names)),np.float32)
        self._local_age_sources={};self.local_gc_bucket_available_ns={}
        for tf in TIMEFRAMES:
            clock=np.asarray(packet['timeframes'][tf]['available_ns'])
            at=np.searchsorted(clock,decisions,side='right')-1
            chosen=np.full(len(decisions),-1,np.int64)
            chosen[at>=0]=clock[at[at>=0]]
            chosen.flags.writeable=False;self.local_gc_bucket_available_ns[tf]=chosen
        for j,field in enumerate(self.base_fields):
            value,observed,source_clock=_field_values(field,_table(packet,field),decisions,include_source=True)
            if field['transform']=='decision_age_minutes':
                source_clock.flags.writeable=False;self._local_age_sources[field['name']]=source_clock
            context[:,2*j]=value;context[:,2*j+1]=observed
        context.flags.writeable=False;self.gc_context=context
        from gx1.contracts.entry_gc_mtf_history_v1 import native_gc_mtf_history_layout
        self.layout=native_gc_mtf_history_layout();self.gc_surfaces={}
        self._side_age_sources={};self._local_tf_join_positions={};self._coherence_columns={}
        for tf in TIMEFRAMES:
            times=self.ta.tf_times[tf];available=times+int(ta.htf.MULTI_TF_SHIFT[tf].value)
            names=self.layout['field_names'][tf];positions={name:i for i,name in enumerate(names)}
            matrix=np.zeros((len(times),len(names)),np.float32)
            self._side_age_sources[tf]={}
            for field in (f for f in self.fields if f['timeframe']==tf):
                value,observed,source_clock=_field_values(field,_table(packet,field),available,exact=True,include_source=True)
                if field['transform']=='decision_age_minutes':
                    source_clock.flags.writeable=False;self._side_age_sources[tf][field['name']]=source_clock
                matrix[:,positions[field['name']]]=value
                # Shared support gates already address an actual structural
                # value; they must never overwrite that original source value.
                if field['availability_field']==field['name']+'_observed':
                    matrix[:,positions[field['availability_field']]]=observed
            matrix.flags.writeable=False;available.flags.writeable=False
            self.gc_surfaces[tf]={'features':matrix,'native_row_indices':self.ta.tf_source_rows[tf],
                'available_ns':available,'field_names':list(names)}
            chosen=self.local_gc_bucket_available_ns[tf]
            join=np.searchsorted(available,chosen)
            inside=np.flatnonzero((chosen>=0)&(join<len(available)))
            good=inside[available[join[inside]]==chosen[inside]]
            aligned=np.full(len(chosen),-1,np.int64);aligned[good]=join[good]
            aligned.flags.writeable=False;self._local_tf_join_positions[tf]=aligned
            columns=[];ages=[]
            for j,field in enumerate(self.base_fields):
                if field['timeframe']!=tf:continue
                columns.append((2*j+1,positions[field['availability_field']]))
                if field['transform']=='decision_age_minutes':ages.append((j,positions[field['name']],field['name']))
                else:columns.append((2*j,positions[field['name']]))
            self._coherence_columns[tf]=(columns,ages)
        self.source_axes=_seal({'schema_version':'gx1_native_m5_full_gc_source_axes_v1',
            'ta_source_axes_sha256':self.ta.source_axes_sha256,
            'native_gc_source_binding_sha256':self.source_binding['contract_sha256'],
            'native_field_catalog_sha256':self.catalog['contract_sha256'],
            'retained_source_rows_sha256':_array_sha(self.retained_source_rows),
            'local_gc_values_sha256':_array_sha(self.gc_context),
            'timeframes':{tf:{'rows':len(s['features']),'features_sha256':_array_sha(s['features']),
                'native_row_indices_sha256':_array_sha(s['native_row_indices']),
                'available_ns_sha256':_array_sha(s['available_ns']),
                'local_current_gc_bucket_available_ns_sha256':_array_sha(self.local_gc_bucket_available_ns[tf]),
                'local_overlapping_TA_global_rows_sha256':_array_sha(np.where(self._local_tf_join_positions[tf]>=0,
                    s['native_row_indices'][np.maximum(0,self._local_tf_join_positions[tf])],-1)),
                'local_age_source_clocks':{k:_array_sha(v) for k,v in self._local_age_sources.items()
                    if k in self._side_age_sources[tf]},
                'own_tf_age_source_clocks':{k:_array_sha(v) for k,v in self._side_age_sources[tf].items()}}
                for tf,s in self.gc_surfaces.items()}})
        self.source_axes_sha256=self.source_axes['contract_sha256']

    def gather(self,source_rows,*,max_rows):
        rows=np.asarray(source_rows);base=self.ta.gather(rows,max_rows=max_rows)
        positions=self.ta._feature_positions(rows)
        ids=positions[:,None]+np.arange(-ta.MODEL_NATIVE_SEQ_LEN+1,1,dtype=np.int64)
        base['seq_x']=np.concatenate((base['seq_x'],self.gc_context[ids]),axis=-1)
        base['snap_x']=np.concatenate((base['snap_x'],self.gc_context[positions]),axis=-1)
        base['ctx_cont']=np.concatenate((base['ctx_cont'],self.gc_context[positions]),axis=-1)
        for tf in TIMEFRAMES:
            take=self.ta.tf_tails[tf][rows,None]+np.arange(-ta.WINDOWS[tf]+1,1,dtype=np.int64)
            surface=self.gc_surfaces[tf]
            _need(np.all(surface['available_ns'][take]<=self.issue_ns[rows,None])
                and np.array_equal(surface['native_row_indices'][take],self.ta.tf_source_rows[tf][take]),'GATHER_TF_IDENTITY')
            base['gc_seq_'+tf.lower()]=np.ascontiguousarray(surface['features'][take])
        self._require_gather_coherence(rows,ids,base)
        _need(len(base)==14 and all(np.isfinite(a).all() for a in base.values()),'JOINT_BATCH')
        return base

    def _require_gather_coherence(self,rows,local_ids,batch):
        width=len(self.gc_context_names)
        _need(np.array_equal(batch['seq_x'][:,-1],batch['snap_x'])
            and np.array_equal(batch['snap_x'][:,-width:],batch['ctx_cont'][:,-width:]),'LOCAL_GC_ALIASES')
        flat=local_ids.reshape(-1)
        for tf in TIMEFRAMES:
            joined=self._local_tf_join_positions[tf][local_ids]
            tails=self.ta.tf_tails[tf][rows,None]
            overlap=(joined>=tails-ta.WINDOWS[tf]+1)&(joined<=tails)&(joined>=0)
            local=flat[overlap.reshape(-1)];side=joined[overlap]
            surface=self.gc_surfaces[tf];columns,ages=self._coherence_columns[tf]
            left=np.asarray([a for a,b in columns],np.int64);right=np.asarray([b for a,b in columns],np.int64)
            for begin in range(0,len(local),512):
                a,b=local[begin:begin+512],side[begin:begin+512]
                _need(np.array_equal(self.gc_context[np.ix_(a,left)],surface['features'][np.ix_(b,right)]),
                      'OVERLAPPING_GC_STATIC:'+tf)
                for field_index,side_column,name in ages:
                    source=self._local_age_sources[name][a]
                    _need(np.array_equal(source,self._side_age_sources[tf][name][b]),'OVERLAPPING_GC_AGE_SOURCE:'+tf)
                    mask=self.gc_context[a,2*field_index+1]==1
                    local_expected=np.where(mask,(self.issue_ns[self.retained_source_rows[a]]-source)/gc.MINUTE_NS,0).astype(np.float32)
                    side_expected=np.where(mask,(surface['available_ns'][b]-source)/gc.MINUTE_NS,0).astype(np.float32)
                    _need(np.all(source[mask]<=surface['available_ns'][b][mask])
                        and np.array_equal(self.gc_context[a,2*field_index],local_expected)
                        and np.array_equal(surface['features'][b,side_column],side_expected),'OVERLAPPING_GC_AGE_VALUE:'+tf)

    def support_report(self,source_rows):
        rows=np.asarray(source_rows,dtype=np.int64)
        positions=self.ta._feature_positions(rows)
        return {'schema_version':'gx1_native_m5_full_gc_feature_support_v1',
            'source_rows':len(rows),'input_ready_rows':int(self.input_ready[rows].sum()),
            'ta':self.ta.support_report(rows),'native_field_catalog_sha256':self.catalog['contract_sha256'],
            'local_gc_field_names':list(self.gc_context_names),
            'local_gc_observed_counts':{f['name']:int(self.gc_context[positions,2*j+1].sum())
                for j,f in enumerate(self.base_fields)},
            'timeframes':{tf:{'rows':len(s['features']),'field_names':s['field_names'],
                'native_row_indices_sha256':_array_sha(s['native_row_indices']),
                'available_ns_sha256':_array_sha(s['available_ns'])} for tf,s in self.gc_surfaces.items()},
            'unknown_gc_rows_removed':0,'source_axes_sha256':self.source_axes_sha256}

    def input_binding(self,normalization):
        from gx1.contracts.entry_model_native_input_normalization_v1 import require_native_m5_full_gc_input_normalization_contract
        contract=require_native_m5_full_gc_input_normalization_contract(normalization)
        history=contract['gc_mtf_history'];provider=native_m5_gc_feature_provider_contract(history)
        _need(contract['source_axes_sha256']==self.source_axes_sha256
            and contract['feature_provider_contract_sha256']==provider['contract_sha256']
            and contract['native_gc_source_binding_sha256']==self.source_binding['contract_sha256'], 'NORMALIZATION_BINDING')
        return require_native_m5_gc_input_binding(_seal({'schema_version':INPUT_SCHEMA,'source_bar_seconds':300,
            'timeframes':list(TIMEFRAMES),'source_clock_contract':CLOCK,
            'feature_provider_contract_sha256':provider['contract_sha256'],
            'source_axes_sha256':self.source_axes_sha256,'input_normalization_sha256':contract['contract_sha256'],
            'execution_profile':provider['execution_profile'],'warmstart_used':False,
            'native_gc_source_binding_sha256':self.source_binding['contract_sha256'],
            'gc_layout_sha256':history['layout']['contract_sha256'],
            'gc_normalization_sha256':history['normalization']['contract_sha256']}))


def fit_native_m5_gc_feature_normalization(provider,*,fit_source_rows,cutoff_ns):
    """Fit all joint inputs on the exact earlier issue/window unions."""
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    from gx1.contracts.entry_gc_mtf_history_v1 import fit_native_m5_gc_history_normalization
    from gx1.features.entry_specialist_feature_groups_v1 import compile_native_m5_full_gc_entry_execution_profile
    _need(type(provider) is NativeM5GCFeatureProvider,'FIT_PROVIDER')
    rows=np.asarray(fit_source_rows)
    _need(rows.dtype==np.int64 and rows.ndim==1 and len(rows)>1 and np.all(np.diff(rows)>0)
        and rows[0]>=0 and rows[-1]<len(provider.issue_ns) and provider.input_ready[rows].all()
        and type(cutoff_ns) is int and 0<cutoff_ns<=NATIVE_M5_TRAIN_FIT_CUTOFF_NS
        and np.all(provider.issue_ns[rows]<cutoff_ns),'FIT_ROWS')
    base=provider.ta
    unions={tf:ta._window_union(base.tf_tails[tf][rows],ta.WINDOWS[tf],len(base.tf_times[tf])) for tf in TIMEFRAMES}
    original={tf:base.tf_source_rows[tf][ids] for tf,ids in unions.items()}
    history_stats=fit_native_m5_gc_history_normalization(provider.gc_surfaces,original,
        train_start_ns=int(provider.issue_ns[rows[0]]),train_end_ns=cutoff_ns,
        source_binding=provider.source_binding,source_axes_sha256=provider.source_axes_sha256,
        layout=provider.layout)
    history={'layout':provider.layout,'normalization':history_stats}
    profile=compile_native_m5_full_gc_entry_execution_profile(history)
    contract=native_m5_gc_feature_provider_contract(history)
    fields=profile['physical_field_names'];surfaces={}
    local=base._feature_positions(ta._window_union(rows,ta.MODEL_NATIVE_SEQ_LEN,len(base.issue_ns)))
    current=base._feature_positions(rows)
    signal=np.concatenate((base.surfaces['signal'][local],provider.gc_context[local]),axis=1)
    surfaces['signal']=norm.fit_native_m5_gc_surface_normalization(signal,surface='signal',
        field_names=fields['signal'],semantic_categorical_domains=norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS)
    del signal
    context=np.concatenate((base.surfaces['ctx_cont'][current],provider.gc_context[current]),axis=1)
    surfaces['ctx_cont']=norm.fit_native_m5_gc_surface_normalization(context,surface='ctx_cont',
        field_names=fields['ctx_cont'],semantic_categorical_domains={})
    owners={i:family for family,indices in profile['local_signal_indices_by_specialist'].items() for i in indices}
    aliases=[{'signal_index':int(si),'ctx_cont_index':int(ci),'signal_field':fields['signal'][si],
        'ctx_cont_field':fields['ctx_cont'][ci],'specialist':owners[si]} for si,ci in zip(
            profile['temporal_alias_signal_indices'],profile['temporal_alias_ctx_cont_indices'])]
    surfaces['ctx_cont']=norm.share_temporal_alias_stats_from_signal(surfaces['ctx_cont'],surfaces['signal'],
        temporal_aliases=aliases,ctx_cont_values=context)
    del context
    population={}
    for tf,ids in unions.items():
        available=base.tf_times[tf][ids]+int(ta.htf.MULTI_TF_SHIFT[tf].value)
        _need(np.all(available<cutoff_ns),'FIT_TF_CUTOFF:'+tf)
        name='mtf_'+tf.lower()
        surfaces[name]=norm.fit_native_m5_surface_normalization(
            norm.MatrixPopulationPart(base.tf_values[tf],row_indices=ids,source=name),
            surface=name,field_names=fields['mtf'],semantic_categorical_domains=norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS)
        population[tf]={'rows':len(ids),'source_rows_sha256':_array_sha(original[tf]),
                        'available_ns_sha256':_array_sha(available)}
    category=norm.fit_ctx_cat_contract(norm.MatrixPopulationPart(base.surfaces['ctx_cat'],row_indices=current,
        source='native_joint_M5_categories'),field_names=fields['ctx_cat'])
    fit={'cutoff_ns':cutoff_ns,'first_issue_ns':int(provider.issue_ns[rows[0]]),
        'last_issue_ns':int(provider.issue_ns[rows[-1]]),'issue_rows':len(rows),
        'selected_issue_ns_sha256':_array_sha(provider.issue_ns[rows]),
        'selected_source_rows_sha256':_array_sha(rows),'local_rows':len(local),'context_rows':len(rows),
        'per_tf':population}
    return norm.build_native_m5_full_gc_input_normalization_contract(
        native_m5_manifest=base.surfaces['source_binding']['native_m5_manifest'],
        feature_provider_contract_sha256=contract['contract_sha256'],source_axes_sha256=provider.source_axes_sha256,
        fit_population=fit,surfaces=surfaces,ctx_cat=category,temporal_aliases=aliases,gc_mtf_history=history,
        native_gc_source_binding_sha256=provider.source_binding['contract_sha256'])


def _diagnostic_scalar_bounds(field):
    """Only scalar source domains, not a reconstruction of a market path."""
    unit,column=field['unit'],field['column']
    fractions={'known_aggressor_fraction','observed_known_aggressor_fraction',
        'observed_coverage_fraction','observed_signed_minute_fraction',
        'observed_known_cumulative_unknown_fraction','observed_profile_source_coverage_fraction',
        'observed_book_flow_coverage_fraction','observed_book_coverage_fraction',
        'observed_order_coverage_fraction','observed_event_coverage_fraction'}
    imbalances={'tw_imbalance','observed_book_tw_imbalance',
        'tw_order_count_imbalance','observed_order_tw_order_count_imbalance'}
    if unit=='ratio':
        if column in fractions:return {'lower':0.,'upper':1.,'lower_inclusive':True}
        if column in imbalances:return {'lower':-1.,'upper':1.,'lower_inclusive':True}
        raise RuntimeError('NATIVE_M5_GC_DIAGNOSTIC_UNDECLARED_BOUNDS:' + field['name'])
    if unit in ('USD','price_tick'):
        return {'lower':0.,'upper':None,'lower_inclusive':False}
    if unit=='basis_point':return {'lower':-10_000.,'upper':None,'lower_inclusive':False}
    if unit in ('event','trade','order','price_level','minute','bar','interval','event_per_second','contract_per_order'):
        return {'lower':0.,'upper':None,'lower_inclusive':True}
    if unit in ('contract','tick','nanosecond'):
        return {'lower':None,'upper':None,'lower_inclusive':False}
    raise RuntimeError('NATIVE_M5_GC_DIAGNOSTIC_UNDECLARED_BOUNDS:' + field['name'])


def _diagnostic_packet_gather(provider,packet,rows,*,max_rows):
    """Use the production source/query owner without creating a new binding."""
    result=provider.ta.gather(rows,max_rows=max_rows)
    positions=provider.ta._feature_positions(rows)
    local=positions[:,None]+np.arange(-ta.MODEL_NATIVE_SEQ_LEN+1,1,dtype=np.int64)
    decisions=provider.issue_ns[provider.retained_source_rows[local.reshape(-1)]]
    context=np.zeros((len(decisions),len(provider.gc_context_names)),np.float32)
    for j,field in enumerate(provider.base_fields):
        value,observed=_field_values(field,_table(packet,field),decisions)
        context[:,2*j]=value;context[:,2*j+1]=observed
    sequence=context.reshape(len(rows),ta.MODEL_NATIVE_SEQ_LEN,-1)
    result['seq_x']=np.concatenate((result['seq_x'],sequence),axis=-1)
    result['snap_x']=np.concatenate((result['snap_x'],sequence[:,-1]),axis=-1)
    result['ctx_cont']=np.concatenate((result['ctx_cont'],sequence[:,-1]),axis=-1)
    for tf in TIMEFRAMES:
        take=provider.ta.tf_tails[tf][rows,None]+np.arange(-ta.WINDOWS[tf]+1,1,dtype=np.int64)
        available=provider.gc_surfaces[tf]['available_ns'][take].reshape(-1)
        names=provider.layout['field_names'][tf];at={name:i for i,name in enumerate(names)}
        features=np.zeros((len(available),len(names)),np.float32)
        for field in (f for f in provider.fields if f['timeframe']==tf):
            value,observed=_field_values(field,_table(packet,field),available,exact=True)
            features[:,at[field['name']]]=value
            if field['availability_field']==field['name']+'_observed':
                features[:,at[field['availability_field']]]=observed
        result['gc_seq_'+tf.lower()]=features.reshape(len(rows),ta.WINDOWS[tf],len(names))
    _need(all(np.isfinite(a).all() for a in result.values()),'DIAGNOSTIC_BATCH_NONFINITE')
    return result


def diagnostic_regather_native_m5_gc_primitive(provider,*,source_packet,input_normalization,
        field_name,source_available_ns,delta,source_rows,max_rows):
    """Return a view-consistent input counterfactual, never a source admission.

    Exactly one numeric packet cell changes. Its same-column views are queried
    again through the production field owner. Sibling indicators remain fixed;
    this does not reconstruct trades, a new market path or economic outcomes.
    No perturbed provider, production binding or COMPLETE receipt is created.
    """
    _need(type(provider) is NativeM5GCFeatureProvider,'DIAGNOSTIC_PROVIDER')
    _need(type(max_rows) is int and 0<max_rows<=16,'DIAGNOSTIC_BATCH_LIMIT')
    fields={field['name']:field for field in provider.fields}
    _need(field_name in fields,'DIAGNOSTIC_FIELD')
    field=fields[field_name]
    _need(not field['binary_value'] and not field['structural_mask_value']
        and field['transform'] is None and field['role']!='structural_observation_mask',
        'DIAGNOSTIC_NONMARKET_FIELD')
    _need(type(source_available_ns) is int and type(delta) in (int,float)
        and np.isfinite(delta) and float(delta)==field['unit_scale'],'DIAGNOSTIC_ONE_SOURCE_UNIT')
    arrays,layout,axes=_packet_layout(source_packet)
    _need(_sha(layout)==provider.source_binding['packet_array_contract_sha256']
        and axes==provider.source_binding['packet_axes']
        and source_packet['source_clock']==provider.source_binding['source_clock'],'DIAGNOSTIC_ORIGINAL_SOURCE')
    table=_table(source_packet,field);clock=np.asarray(table['available_ns'])
    selected=np.flatnonzero(clock==source_available_ns)
    _need(len(selected)==1,'DIAGNOSTIC_SOURCE_CLOCK')
    index=int(selected[0]);original_value,observed=_field_values(field,table,np.array([source_available_ns],np.int64),exact=True)
    _need(bool(observed[0]),'DIAGNOSTIC_UNOBSERVED_SOURCE')
    raw=np.asarray(table[field['column']]);old=float(raw[index]);new=old+float(delta)
    bounds=_diagnostic_scalar_bounds(field)
    for value in (old,new):
        _need(np.isfinite(value),'DIAGNOSTIC_NONFINITE_PRIMITIVE')
        if bounds['lower'] is not None:
            _need(value>=bounds['lower'] if bounds['lower_inclusive'] else value>bounds['lower'],
                  'DIAGNOSTIC_SOURCE_BOUNDS')
        if bounds['upper'] is not None:_need(value<=bounds['upper'],'DIAGNOSTIC_SOURCE_BOUNDS')
    rows=np.asarray(source_rows)
    before=provider.gather(rows,max_rows=max_rows)
    _need(source_available_ns<=int(provider.issue_ns[rows].max()),'DIAGNOSTIC_FUTURE_SOURCE')
    input_binding=provider.input_binding(input_normalization)
    norm_before=_sha(input_normalization);binding_before=_sha(provider.source_binding)
    proof=dict(provider.source_binding['assembly_proof']);_file_ref(proof)
    rebuilt=_diagnostic_packet_gather(provider,source_packet,rows,max_rows=max_rows)
    _need(all(np.array_equal(before[k].view(np.uint8),rebuilt[k].view(np.uint8)) for k in before),
          'DIAGNOSTIC_ORIGINAL_REGATHER')
    changed_column=raw.copy();changed_column[index]=new
    _need(float(changed_column[index])==new and float(changed_column[index])!=old,'DIAGNOSTIC_SOURCE_PRECISION')
    variant=dict(source_packet)
    if field['packet_group']=='timeframes':
        variant['timeframes']=dict(source_packet['timeframes'])
        variant['timeframes'][field['timeframe']]=dict(table)
        variant['timeframes'][field['timeframe']][field['column']]=changed_column
    else:
        variant['m5_book_features']=dict(table);variant['m5_book_features'][field['column']]=changed_column
    after=_diagnostic_packet_gather(provider,variant,rows,max_rows=max_rows)
    primitive=(field['packet_group'],field['timeframe'],field['column'])
    affected=[item for item in provider.fields if (item['packet_group'],item['timeframe'],item['column'])==primitive]
    positions=provider.ta._feature_positions(rows)
    local=positions[:,None]+np.arange(-ta.MODEL_NATIVE_SEQ_LEN+1,1,dtype=np.int64)
    local_clock=provider.issue_ns[provider.retained_source_rows[local]]
    def selected_at(decisions,exact=False):
        at=np.searchsorted(clock,decisions,side='right')-1
        chosen=at==index
        if exact:chosen&=np.where(at>=0,clock[np.maximum(at,0)],-1)==decisions
        return chosen
    local_selected=selected_at(local_clock)
    current_selected=local_selected[:,-1]
    base_positions={item['name']:i for i,item in enumerate(provider.base_fields)}
    changed_counts={};changed_coordinates={};batch_hashes={'before':{},'after':{}}
    for key in before:
        original,modified=before[key],after[key]
        _need(original.shape==modified.shape and original.dtype==modified.dtype,'DIAGNOSTIC_BATCH_HEADER')
        allowed=np.zeros(original.shape,bool)
        for item in affected:
            if item['name'] in base_positions:
                column=2*base_positions[item['name']]
                if key=='seq_x':allowed[:,:,provider.ta.surfaces['signal'].shape[1]+column]|=local_selected
                elif key=='snap_x':allowed[:,provider.ta.surfaces['signal'].shape[1]+column]|=current_selected
                elif key=='ctx_cont':allowed[:,provider.ta.surfaces['ctx_cont'].shape[1]+column]|=current_selected
            if key=='gc_seq_'+item['timeframe'].lower():
                tf=item['timeframe']
                take=provider.ta.tf_tails[tf][rows,None]+np.arange(-ta.WINDOWS[tf]+1,1,dtype=np.int64)
                decision=provider.gc_surfaces[tf]['available_ns'][take]
                column=provider.layout['field_names'][tf].index(item['name'])
                allowed[:,:,column]|=selected_at(decision,exact=True)
        difference=np.any(original.view(np.uint8).reshape(*original.shape,original.dtype.itemsize)
            !=modified.view(np.uint8).reshape(*modified.shape,modified.dtype.itemsize),axis=-1)
        _need(not np.any(difference&~allowed),'DIAGNOSTIC_UNRELATED_COORDINATE:'+key)
        changed_counts[key]=int(difference.sum())
        changed_coordinates[key]=np.argwhere(difference).tolist()
        batch_hashes['before'][key]=_array_sha(original);batch_hashes['after'][key]=_array_sha(modified)
    _need(sum(changed_counts.values())>0,'DIAGNOSTIC_SOURCE_NOT_CONSUMED')
    width=len(provider.gc_context_names)
    _need(np.array_equal(after['seq_x'][:,-1],after['snap_x'])
        and np.array_equal(after['snap_x'][:,-width:],after['ctx_cont'][:,-width:]),'DIAGNOSTIC_LOCAL_ALIASES')
    # Every matched own-TF view reads precisely the same changed packet scalar.
    for item in affected:
        if item['name'] not in base_positions:continue
        tf=item['timeframe'];column=provider.layout['field_names'][tf].index(item['name'])
        take=provider.ta.tf_tails[tf][rows,None]+np.arange(-ta.WINDOWS[tf]+1,1,dtype=np.int64)
        chosen=selected_at(provider.gc_surfaces[tf]['available_ns'][take],exact=True)
        if chosen.any():_need(np.all(after['gc_seq_'+tf.lower()][:,:,column][chosen]==np.float32(new)),
                             'DIAGNOSTIC_OWN_TF_ALIAS')
    _,final_layout,final_axes=_packet_layout(source_packet)
    _need(final_layout==layout and final_axes==axes and _sha(input_normalization)==norm_before
        and _sha(provider.source_binding)==binding_before,'DIAGNOSTIC_ORIGINAL_MUTATED')
    _file_ref(proof)
    receipt=_seal({'schema_version':'gx1_native_m5_gc_input_primitive_diagnostic_v1',
        'purpose':'DIAGNOSTIC_INPUT_PRIMITIVE_COUNTERFACTUAL','source_admission':False,
        'market_path_reconstructed':False,'sibling_indicator_formulas_recomputed':False,
        'production_binding_created':False,'model_weights_modified':False,
        'original_source_binding_sha256':provider.source_binding['contract_sha256'],
        'original_source_axes_sha256':provider.source_axes_sha256,'original_assembly_proof':proof,
        'input_normalization_sha256':input_binding['input_normalization_sha256'],
        'original_source_arrays_unchanged':True,'original_proof_bytes_unchanged':True,
        'original_normalization_bytes_unchanged':True,
        'primitive':{'field_name':field_name,'packet_group':primitive[0],'timeframe':primitive[1],
            'column':primitive[2],'source_index':index,'source_row_available_ns':source_available_ns,
            'field_source_available_ns':int(np.asarray(table[field['source_clock_column']])[index]),
            'before':old,'after':new,'delta':float(delta),'unit':field['unit'],
            'unit_scale':field['unit_scale'],'unit_owner':field['unit_owner'],'scalar_bounds':bounds},
        'affected_fields':[item['name'] for item in affected],
        'source_rows':[int(row) for row in rows],'changed_coordinate_counts':changed_counts,
        'changed_coordinates':changed_coordinates,'batch_sha256':batch_hashes})
    return {'before':before,'after':after,'diagnostic':receipt}
