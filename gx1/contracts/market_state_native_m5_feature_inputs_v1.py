"""Native M5-only technical/macro values, physical populations and causal batches.

No label, former Entry axis, M1 provider, checkpoint or old fitted statistic is
accepted here. Missing history remains NaN in the physical surfaces. Only rows
whose actual local and five-TF windows are observed can enter a model batch.
GC MBP inputs have their separate retained owner and are outside the full-TA
profile; this module never fabricates them for earlier history.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_gc_mtf_history_v1 import TEST_BOUNDARY_NS
from gx1.features import htf_features as htf
from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import entry_model_native_feature_layers_v1 as layers
from gx1.features import d1_macro_context_v1 as macro
from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_DIRECTION_CLOCKS as TIMEFRAMES,
    VOLATILITY_SQUEEZE_DIRECTION_CLOCK_CONTRACT as CLOCK_CONTRACT,
    require_volatility_squeeze_artifact_set, VolatilitySqueezeArtifactSet,
)

BAR_NS = 300_000_000_000
WINDOWS = dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)
RAW_COLUMNS = ('open','high','low','close','bid_open','bid_high','bid_low','bid_close',
               'ask_open','ask_high','ask_low','ask_close','volume')
BINDING_SCHEMA = 'gx1_native_m5_bar_direction_inputs_v1'
SOURCE_SCHEMA = 'gx1_native_m5_feature_source_v1'
SURFACES_SCHEMA = 'gx1_native_m5_full_ta_surfaces_v1'


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('NATIVE_M5_FEATURE_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_sha(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _seal(value):
    return {**value, 'contract_sha256': _sha(value)}


def _sealed(value, reason):
    _need(isinstance(value, Mapping) and value.get('contract_sha256') == _sha(
        {k:v for k,v in value.items() if k != 'contract_sha256'}), reason)


def _ref(value):
    _need(isinstance(value, Mapping) and set(value) == {'path','sha256'}, 'REF')
    p, digest = value['path'], value['sha256']
    _need(isinstance(p,str) and Path(p).is_absolute() and '..' not in Path(p).parts
          and str(Path(p)) == p and isinstance(digest,str) and len(digest) == 64
          and all(c in '0123456789abcdef' for c in digest), 'REF')
    return dict(value)


def _owned(value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def native_m5_feature_provider_contract():
    profile = owners.compile_full_ta_entry_execution_profile()
    return _seal({
        'schema_version':'gx1_native_m5_feature_provider_v1',
        'source_clock_contract':CLOCK_CONTRACT, 'source_bar_seconds':300,
        'execution_profile':{'name':profile['name'],'contract_sha256':profile['contract_sha256']},
        'timeframes':list(TIMEFRAMES), 'field_names':{k:list(v) for k,v in profile['physical_field_names'].items()},
        'local_sequence_rows':MODEL_NATIVE_SEQ_LEN, 'window_rows':dict(WINDOWS),
        'mtf_matrix_contract':htf.HTF_V4_MATRIX_CONTRACT,
        'local_price_formula_contract':layers.price_derived_contract_metadata(),
        'local_value_owners':{'same_M5_exact_aliases':[list(p) for p in layers.model_native_m5_local_per_bar_exact_aliases()],
            'local_trendline':'gx1.features.trendline_registry_v1.compute_trendline_registry_features_v1',
            'closed_context':'gx1.features.htf_features', 'session':'gx1.features.basic_v1.add_session_features',
            'calendar':'gx1.scripts.materialize_canonical_v3_augment.add_cyclic_time_features'},
        'macro_contract':macro.d1_macro_context_contract(),
        'missing_values':'physical_NaN_and_explicit_observed_arrays_never_imputed',
        'input_selection':'calibration_known_at_issue_and_observed_local_and_five_TF_windows_only_no_target_metadata',
        'gc_scope':'separate_retained_GC_owner_not_in_full_TA_profile',
    })


def require_native_m5_input_binding(value):
    _sealed(value,'INPUT_BINDING_HASH')
    _need(set(value) == {'schema_version','source_bar_seconds','timeframes','source_clock_contract',
        'feature_provider_contract_sha256','source_axes_sha256','input_normalization_sha256',
        'execution_profile','warmstart_used','contract_sha256'}, 'INPUT_BINDING_KEYS')
    contract = native_m5_feature_provider_contract()
    _need(value['schema_version'] == BINDING_SCHEMA and value['source_bar_seconds'] == 300
          and value['timeframes'] == list(TIMEFRAMES) and value['source_clock_contract'] == CLOCK_CONTRACT
          and value['execution_profile'] == contract['execution_profile']
          and value['feature_provider_contract_sha256'] == contract['contract_sha256']
          and value['warmstart_used'] is False, 'INPUT_BINDING_IDENTITY')
    for key in ('source_axes_sha256','input_normalization_sha256'):
        digest=value[key]
        _need(isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'INPUT_BINDING_DIGEST')
    return json.loads(json.dumps(value))


def _raw_frame(frame):
    _need(isinstance(frame,pd.DataFrame) and isinstance(frame.index,pd.DatetimeIndex)
          and frame.index.tz is not None and str(frame.index.tz)=='UTC'
          and tuple(frame.columns)==RAW_COLUMNS and len(frame)>0, 'RAW_FRAME')
    starts=frame.index.as_unit('ns').asi8
    _need(np.all(starts>=0) and np.all(starts % BAR_NS==0) and np.all(np.diff(starts)>0),'RAW_CLOCK')
    values=frame.loc[:,list(RAW_COLUMNS)].to_numpy(np.float64)
    _need(np.isfinite(values).all() and np.all(values[:,:12]>0)
          and np.all(values[:,-1]>=0) and np.all(values[:,-1]==np.floor(values[:,-1])),'RAW_VALUES')
    for prefix in ('','bid_','ask_'):
        op,hi,lo,cl=(frame[prefix+n].to_numpy(np.float64) for n in ('open','high','low','close'))
        _need(np.all(hi>=np.maximum(op,cl)) and np.all(lo<=np.minimum(op,cl)),'RAW_OHLC')
    return starts,values


def bind_native_m5_feature_source(native_m5, *, native_m5_manifest, source_spec):
    """Bind caller-decoded native rows to one original manifest/part inventory.

    The caller's capped producer verifies the declared raw-part hashes before
    decoding. This owner verifies the original native manifest, the exact part
    bindings, raw clock/cutoff and every supplied price/quote/volume byte.
    """
    manifest_ref=_ref(native_m5_manifest)
    _need(isinstance(source_spec,Mapping) and set(source_spec)=={'manifest','parts','cutoff_ns'}
          and source_spec['manifest']==manifest_ref and type(source_spec['cutoff_ns']) is int
          and 0<source_spec['cutoff_ns']<=TEST_BOUNDARY_NS,'SOURCE_SPEC')
    path=Path(manifest_ref['path'])
    _need(path.name=='MANIFEST.json' and not path.is_symlink() and path.is_file()
          and path.resolve(strict=True)==path,'NATIVE_MANIFEST_PATH')
    raw=path.read_bytes()
    _need(hashlib.sha256(raw).hexdigest()==manifest_ref['sha256'],'NATIVE_MANIFEST_HASH')
    manifest=json.loads(raw)
    required={'schema_version':'xau_canonical_native_source_v3','instrument':'XAU_USD','timeframe':'M5',
        'source_granularity':'M5','source_kind':'oanda_native_mba_candles','prices':'MBA',
        'bar_duration_seconds':300,'decision_available_offset_seconds':300,'timestamp_semantics':'bar_start_utc',
        'completion_field':'complete','completion_value':True}
    _need(all(manifest.get(k)==v for k,v in required.items()),'NATIVE_MANIFEST_IDENTITY')
    parts=source_spec['parts'];_need(isinstance(parts,list) and parts,'SOURCE_PARTS')
    names=[]
    for part in parts:
        _need(isinstance(part,Mapping) and set(part)=={'path','sha256','row_groups'},'SOURCE_PART')
        p=Path(str(part['path']));groups=part['row_groups']
        _need(p==path.parent/p.parent.name/'part-000.parquet'
              and part['sha256']==manifest['year_sha256'].get(p.parent.name)
              and isinstance(groups,list) and groups and all(type(i) is int and i>=0 for i in groups)
              and groups==sorted(set(groups)),'SOURCE_PART_IDENTITY')
        names.append(str(p))
    _need(names==sorted(set(names)),'SOURCE_PART_ORDER')
    starts,values=_raw_frame(native_m5)
    _need(np.all(starts<source_spec['cutoff_ns']-BAR_NS),'SOURCE_EXCLUSIVE_CLOSE_CUTOFF')
    return _seal({'schema_version':SOURCE_SCHEMA,'native_m5_manifest':manifest_ref,
        'source_clock_contract':CLOCK_CONTRACT,'source_spec':json.loads(json.dumps(source_spec)),
        'rows':len(starts),'bar_start_ns_sha256':_array_sha(starts,'<i8'),
        'raw_columns':list(RAW_COLUMNS),'raw_values_sha256':_array_sha(values,'<f8')})


def build_native_m5_feature_surfaces(native_m5, *, source_binding, v29_registry_constants,
                                      volatility_squeeze_artifacts):
    """Compute all full-TA fields from native M5 and declared as-of macro sources."""
    _sealed(source_binding,'SOURCE_BINDING_HASH')
    _need(source_binding.get('schema_version')==SOURCE_SCHEMA
          and source_binding.get('source_clock_contract')==CLOCK_CONTRACT,'SOURCE_BINDING_IDENTITY')
    rebound=bind_native_m5_feature_source(native_m5,native_m5_manifest=source_binding['native_m5_manifest'],
                                         source_spec=source_binding['source_spec'])
    _need(rebound==source_binding,'RAW_SOURCE_REBIND')
    _need(isinstance(volatility_squeeze_artifacts,VolatilitySqueezeArtifactSet)
          and volatility_squeeze_artifacts.clocks==TIMEFRAMES,'NATIVE_FIVE_SQUEEZE_REQUIRED')
    registry=htf.require_native_m5_registry_constants(v29_registry_constants)
    squeeze=require_volatility_squeeze_artifact_set(volatility_squeeze_artifacts)
    registry_source=registry['provenance']['entry_m5_trendline_band']['source_provenance']
    squeeze_params=squeeze.params_by_clock['M5']
    squeeze_source=squeeze_params['source_provenance']
    expected_native=source_binding['native_m5_manifest']
    for provenance in (registry_source,squeeze_source):
        _need({'path':provenance['native_m5_manifest_artifact'],
               'sha256':provenance['native_m5_manifest_sha256']}==expected_native,'CALIBRATION_NATIVE_SOURCE_JOIN')
    _need(all(registry[k]==squeeze_params[k] for k in
        ('declared_train_window_start','declared_train_window_end')),'CALIBRATION_WINDOW_JOIN')
    parameter_available_ns=int(pd.Timestamp(registry['declared_train_window_end']).value)
    _need(parameter_available_ns<source_binding['source_spec']['cutoff_ns'],'CALIBRATION_SOURCE_CUTOFF')
    source=native_m5.copy(deep=False);source.index=source.index.as_unit('ns')
    multi_tf=htf.build_multi_tf_per_bar_features_v4(source,v29_registry_constants=registry,
                                                  volatility_squeeze_artifacts=squeeze)
    # One exact same-clock alias owner; no copying between resolutions.
    m5=multi_tf['M5'];_need(m5.index.equals(source.index),'LOCAL_M5_AXIS')
    values={};lineage={}
    for local,per_bar in layers.model_native_m5_local_per_bar_exact_aliases():
        values[local]=m5[per_bar].to_numpy(np.float32,copy=False)
        lineage[local]='same_native_M5_owner:'+per_bar
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1
    trend_source=source.loc[:,['high','low','close']].copy()
    trend_source['atr']=htf._atr(trend_source['high'],trend_source['low'],trend_source['close'],14)
    trend,_state=compute_trendline_registry_features_v1(trend_source,timeframe='M5',
        seq_len=int(registry['entry_m5']['seq_len']),band_atr=float(registry['entry_m5']['trendline_band_atr']))
    _need(len(trend.columns)==len(layers.TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES),'LOCAL_TRENDLINE_FIELDS')
    for local,name in zip(layers.TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,trend.columns):
        values[local]=trend[name].to_numpy(np.float32);lineage[local]='local_native_M5_trendline:'+name
    context=source.copy(deep=False)
    htf.attach_model_native_mtf_scalars_v4(context,multi_tf=multi_tf,decision_bar_duration=pd.Timedelta(minutes=5))
    regime=htf.project_multi_tf_v4_scalars(multi_tf,source.index.asi8,
        (('trend_state_age_bars','trend_state_age_bars'),('ema_stack_aligned','ema_stack_aligned_v2')),
        tuple(tf.lower() for tf in TIMEFRAMES),decision_bar_duration=pd.Timedelta(minutes=5))
    for name,vector in regime.items():context[name]=vector
    from gx1.features.basic_v1 import add_session_features
    from gx1.scripts.materialize_canonical_v3_augment import add_cyclic_time_features
    context=add_cyclic_time_features(add_session_features(context,decision_delay_seconds=300))
    macro_spec=macro.load_d1_macro_context_source_spec(Path(__file__).resolve().parents[2])
    macro_components=macro.d1_macro_context_components(macro_spec)
    macro_frame=macro.build_d1_macro_context(source.index+pd.Timedelta(minutes=5),macro_components)
    for name in macro.D1_MACRO_CONTEXT_FIELDS:context[name]=macro_frame[name].to_numpy(np.float64)
    profile=owners.compile_full_ta_entry_execution_profile();fields=profile['physical_field_names']
    for name in fields['signal']:
        if name not in values:
            _need(name.startswith('ctx_cont.') and name[9:] in context,'UNOWNED_FIELD:'+name)
            values[name]=context[name[9:]].to_numpy(np.float32)
            lineage[name]=('asof_macro:'+name[9:] if name[9:] in macro.D1_MACRO_CONTEXT_FIELDS
                           else 'closed_context_or_session:'+name[9:])
    signal=np.column_stack([values[n] for n in fields['signal']]).astype(np.float32)
    ctx=np.empty((len(signal),len(fields['ctx_cont'])),dtype=np.float32)
    signal_aliases=list(profile['temporal_alias_signal_indices'])
    context_aliases=list(profile['temporal_alias_ctx_cont_indices'])
    ctx[:,context_aliases]=signal[:,signal_aliases]
    _need(all(fields['signal'][si]=='ctx_cont.'+fields['ctx_cont'][ci]
              for si,ci in zip(signal_aliases,context_aliases)),'CONTEXT_ALIAS_ORDER')
    cat=context.loc[:,list(fields['ctx_cat'])].to_numpy(np.int64)
    _need(not np.isinf(signal).any(),'INFINITE_FEATURE')
    starts=source.index.asi8
    macro_binding=macro.d1_macro_context_binding(macro_spec,macro_frame)
    squeeze_binding=squeeze.binding()
    for frame in multi_tf.values():
        frame.attrs['feats_np'].flags.writeable=False
        frame.attrs['ts_int64'].flags.writeable=False
    axes=_seal({'source_binding_sha256':source_binding['contract_sha256'],
        'registry_contract_sha256':registry['contract_sha256'],
        'squeeze_binding_sha256':_sha(squeeze_binding),'macro_binding_sha256':macro_binding['binding_sha256'],
        'parameter_available_ns':parameter_available_ns,
        'rows':len(starts),'source_row_sha256':_array_sha(np.arange(len(starts)),'<i8'),
        'bar_start_ns_sha256':_array_sha(starts,'<i8'),'issue_ns_sha256':_array_sha(starts+BAR_NS,'<i8')})
    return {'schema_version':SURFACES_SCHEMA,'source_binding':source_binding,
        'provider_contract':native_m5_feature_provider_contract(),'axes':axes,
        'bar_start_ns':_owned(starts,np.int64),'signal':_owned(signal,np.float32),
        'ctx_cont':_owned(ctx,np.float32),'ctx_cat':_owned(cat,np.int64),'multi_tf_sources':multi_tf,
        'observed':_owned(np.isfinite(signal),np.bool_),
        'field_lineage':{name:lineage[name] for name in fields['signal']},
        'registry_contract_sha256':registry['contract_sha256'],'squeeze_binding':squeeze_binding,
        'macro_binding':macro_binding,'parameter_available_ns':parameter_available_ns}


class NativeM5FeatureProvider:
    """Exact raw forward inputs for closed M5 rows, without target-derived axes."""
    def __init__(self,surfaces):
        _need(isinstance(surfaces,Mapping) and surfaces.get('schema_version')==SURFACES_SCHEMA,'SURFACES')
        _need(surfaces['provider_contract']==native_m5_feature_provider_contract(),'PROVIDER_IDENTITY')
        self.surfaces=dict(surfaces);self.bar_start_ns=surfaces['bar_start_ns']
        self.issue_ns=_owned(self.bar_start_ns+BAR_NS,np.int64)
        self.source_row=_owned(np.arange(len(self.issue_ns)),np.int64)
        _sealed(surfaces['source_binding'],'SOURCE_BINDING_HASH')
        _sealed(surfaces['axes'],'SOURCE_AXES_HASH')
        _need(surfaces['axes']['rows']==len(self.issue_ns)
              and surfaces['axes']['source_row_sha256']==_array_sha(self.source_row,'<i8')
              and surfaces['axes']['bar_start_ns_sha256']==_array_sha(self.bar_start_ns,'<i8')
              and surfaces['axes']['issue_ns_sha256']==_array_sha(self.issue_ns,'<i8')
              and np.all(np.diff(self.issue_ns)>0)
              and surfaces['axes']['source_binding_sha256']==surfaces['source_binding']['contract_sha256']
              and surfaces['axes']['registry_contract_sha256']==surfaces['registry_contract_sha256']
              and surfaces['axes']['squeeze_binding_sha256']==_sha(surfaces['squeeze_binding'])
              and surfaces['axes']['macro_binding_sha256']==surfaces['macro_binding']['binding_sha256']
              and surfaces['axes']['parameter_available_ns']==surfaces['parameter_available_ns'],
              'SOURCE_AXES_VALUES')
        profile=owners.compile_full_ta_entry_execution_profile();self.profile=profile
        fields=profile['physical_field_names'];n=len(self.issue_ns)
        for key,dtype in (('signal',np.float32),('ctx_cont',np.float32),('ctx_cat',np.int64)):
            a=surfaces[key];_need(isinstance(a,np.ndarray) and a.dtype==dtype
                and a.shape==(n,len(fields[key])) and not a.flags.writeable,'SURFACE_HEADER:'+key)
        _need(np.array_equal(surfaces['signal'][:,list(profile['temporal_alias_signal_indices'])],
                            surfaces['ctx_cont'][:,list(profile['temporal_alias_ctx_cont_indices'])],equal_nan=True),'CONTEXT_ALIAS_VALUES')
        _need(np.array_equal(surfaces['observed'],np.isfinite(surfaces['signal'])),'OBSERVED_VALUES')
        for j,name in enumerate(fields['ctx_cat']):
            _need(np.isin(surfaces['ctx_cat'][:,j],norm.CTX_CAT_DOMAINS[name]).all(),'CONTEXT_CATEGORY')
        observed=np.isfinite(surfaces['signal']).all(axis=1)
        bad=np.r_[0,np.cumsum(~observed,dtype=np.int64)]
        end=np.arange(n)+1;begin=np.maximum(0,end-MODEL_NATIVE_SEQ_LEN)
        self._readiness={'current_features_observed':_owned(observed,np.bool_),
            'local_history':_owned(end>=MODEL_NATIVE_SEQ_LEN,np.bool_),
            'local_window_observed':_owned(bad[end]==bad[begin],np.bool_)}
        ready=(end>=MODEL_NATIVE_SEQ_LEN)&(bad[end]==bad[begin])
        self.tf_tails={};self.tf_values={};self.tf_times={}
        _need(set(surfaces['multi_tf_sources'])==set(TIMEFRAMES),'FIVE_TF')
        for tf in TIMEFRAMES:
            frame=surfaces['multi_tf_sources'][tf]
            _need(list(frame.columns)==list(fields['mtf']),'TF_FIELDS:'+tf)
            times=np.asarray(frame.attrs['ts_int64'],dtype=np.int64)
            matrix=np.asarray(frame.attrs['feats_np'],dtype=np.float32)
            _need(np.array_equal(frame.index.asi8,times) and matrix.shape==(len(times),len(fields['mtf']))
                  and np.all(np.diff(times)>0) and not np.isinf(matrix).any()
                  and not matrix.flags.writeable and not times.flags.writeable,'TF_HEADER:'+tf)
            tail=np.searchsorted(times,self.issue_ns-int(htf.MULTI_TF_SHIFT[tf].value),side='right')-1
            length=WINDOWS[tf];safe=np.clip(tail,0,len(times)-1)
            finite=np.isfinite(matrix).all(axis=1);counts=np.r_[0,np.cumsum(~finite,dtype=np.int64)]
            left=np.maximum(0,safe-length+1)
            history=tail>=length-1;window=counts[safe+1]==counts[left]
            self._readiness['mtf_'+tf.lower()+'_history']=_owned(history,np.bool_)
            self._readiness['mtf_'+tf.lower()+'_window_observed']=_owned(window,np.bool_)
            ready&=history&window
            self.tf_tails[tf]=_owned(tail,np.int64);self.tf_values[tf]=matrix;self.tf_times[tf]=times
        parameters=self.issue_ns>=surfaces['parameter_available_ns']
        self._readiness['parameters_available']=_owned(parameters,np.bool_)
        ready&=parameters
        self.input_ready=_owned(ready,np.bool_)
        self.source_axes_sha256=surfaces['axes']['contract_sha256']

    def gather(self,source_rows,*,max_rows):
        rows=np.asarray(source_rows)
        _need(type(max_rows) is int and 0<max_rows<=2048 and rows.dtype==np.int64
              and rows.ndim==1 and 0<len(rows)<=max_rows and np.all(rows>=0)
              and np.all(rows<len(self.issue_ns)) and len(np.unique(rows))==len(rows),'BATCH_ROWS')
        _need(self.input_ready[rows].all(),'BATCH_NOT_OBSERVED')
        ids=rows[:,None]+np.arange(-MODEL_NATIVE_SEQ_LEN+1,1,dtype=np.int64)
        out={'seq_x':np.ascontiguousarray(self.surfaces['signal'][ids]),
             'snap_x':np.ascontiguousarray(self.surfaces['signal'][rows]),
             'ctx_cont':np.ascontiguousarray(self.surfaces['ctx_cont'][rows]),
             'ctx_cat':np.ascontiguousarray(self.surfaces['ctx_cat'][rows])}
        for tf in TIMEFRAMES:
            take=self.tf_tails[tf][rows,None]+np.arange(-WINDOWS[tf]+1,1,dtype=np.int64)
            _need(np.all(self.tf_times[tf][take]+int(htf.MULTI_TF_SHIFT[tf].value)<=self.issue_ns[rows,None]),'CAUSAL_TF:'+tf)
            out['seq_'+tf.lower()]=np.ascontiguousarray(self.tf_values[tf][take])
        _need(all(np.isfinite(a).all() for a in out.values()),'BATCH_NONFINITE')
        return out

    def readiness_report(self,source_rows):
        """Reuse the actual readiness masks; reasons overlap and are not summed."""
        rows=np.asarray(source_rows)
        _need(rows.dtype==np.int64 and rows.ndim==1 and len(rows)>0
              and np.all(np.diff(rows)>0) and rows[0]>=0 and rows[-1]<len(self.issue_ns),
              'READINESS_ROWS')
        missing=[]
        names=self.profile['physical_field_names']['signal']
        observed=self.surfaces['observed']
        for j,name in enumerate(names):
            count=int((~observed[rows,j]).sum())
            if count:missing.append({'field':name,'unobserved_current_rows':count})
        return {'rows':len(rows),'input_ready':int(self.input_ready[rows].sum()),
            'input_not_ready':int((~self.input_ready[rows]).sum()),
            'failed_conditions':{name:int((~mask[rows]).sum()) for name,mask in self._readiness.items()},
            'reason_counts_overlap':True,'current_field_missingness':missing,
            'parameter_available_ns':self.surfaces['parameter_available_ns']}

    def support_report(self,source_rows):
        """Actual consumed physical rows and field counts; no usefulness claim."""
        rows=np.asarray(source_rows)
        _need(rows.dtype==np.int64 and rows.ndim==1 and len(rows)>0
              and np.all(np.diff(rows)>0) and rows[0]>=MODEL_NATIVE_SEQ_LEN-1 and rows[-1]<len(self.issue_ns),
              'SUPPORT_ROWS')
        fields=self.profile['physical_field_names']
        ids=_window_union(rows,MODEL_NATIVE_SEQ_LEN,len(self.issue_ns))
        report={}
        for name,matrix,take,names in (
            ('signal',self.surfaces['signal'],ids,fields['signal']),
            ('ctx_cont',self.surfaces['ctx_cont'],rows,fields['ctx_cont'])):
            counts=[int(np.isfinite(matrix[take,j]).sum()) for j in range(matrix.shape[1])]
            report[name]={'physical_rows':len(take),'field_names':list(names),'observed_counts':counts}
        for tf in TIMEFRAMES:
            tails=self.tf_tails[tf][rows]
            _need(np.all(tails>=WINDOWS[tf]-1),'SUPPORT_TF_HISTORY:'+tf)
            ids=_window_union(tails,WINDOWS[tf],len(self.tf_times[tf]))
            matrix=self.tf_values[tf]
            report['mtf_'+tf.lower()]={'physical_rows':len(ids),'field_names':list(fields['mtf']),
                'observed_counts':[int(np.isfinite(matrix[ids,j]).sum()) for j in range(matrix.shape[1])]}
        return {'source_rows':len(rows),'input_ready_rows':int(self.input_ready[rows].sum()),
            'surfaces':report,'families':list(self.profile['local_specialists']),
            'macro_fields':list(self.profile['macro_context_fields']),
            'field_lineage':dict(self.surfaces['field_lineage']),
            'gc_scope':self.surfaces['provider_contract']['gc_scope']}

    def input_binding(self,normalization):
        contract=norm.require_native_m5_input_normalization_contract(normalization)
        _need(contract['source_axes_sha256']==self.source_axes_sha256
              and contract['feature_provider_contract_sha256']==self.surfaces['provider_contract']['contract_sha256']
              and contract['native_m5_manifest']==self.surfaces['source_binding']['native_m5_manifest'],'NORMALIZATION_BINDING')
        return require_native_m5_input_binding(_seal({'schema_version':BINDING_SCHEMA,'source_bar_seconds':300,
            'timeframes':list(TIMEFRAMES),'source_clock_contract':CLOCK_CONTRACT,
            'feature_provider_contract_sha256':self.surfaces['provider_contract']['contract_sha256'],
            'source_axes_sha256':self.source_axes_sha256,'input_normalization_sha256':contract['contract_sha256'],
            'execution_profile':self.surfaces['provider_contract']['execution_profile'],'warmstart_used':False}))


def _window_union(tails,length,rows):
    delta=np.zeros(rows+1,dtype=np.int64)
    np.add.at(delta,tails-length+1,1);np.add.at(delta,tails+1,-1)
    return np.flatnonzero(np.cumsum(delta[:-1])>0)


def fit_native_m5_feature_normalization(provider,*,fit_source_rows,cutoff_ns):
    """Fresh statistics on exactly the caller-declared earlier fit issues.

    The separate target owner proves maturity before supplying this population;
    no target array, future mask or future time is accepted as a model feature.
    """
    _need(type(provider) is NativeM5FeatureProvider,'FIT_PROVIDER')
    rows=np.asarray(fit_source_rows)
    _need(rows.dtype==np.int64 and rows.ndim==1 and len(rows)>1 and np.all(np.diff(rows)>0)
          and rows[0]>=0 and rows[-1]<len(provider.issue_ns) and provider.input_ready[rows].all()
          and type(cutoff_ns) is int and 0<cutoff_ns<=norm.NATIVE_M5_TRAIN_FIT_CUTOFF_NS
          and np.all(provider.issue_ns[rows]<cutoff_ns),'FIT_POPULATION')
    fields=provider.profile['physical_field_names'];raw=provider.surfaces
    local=_window_union(rows,MODEL_NATIVE_SEQ_LEN,len(provider.issue_ns))
    parts={'signal':norm.MatrixPopulationPart(raw['signal'],row_indices=local,source='native_M5_local'),
           'ctx_cont':norm.MatrixPopulationPart(raw['ctx_cont'],row_indices=rows,source='native_M5_context')}
    aliases=[{'signal_index':int(si),'ctx_cont_index':int(ci),'signal_field':fields['signal'][si],
              'ctx_cont_field':fields['ctx_cont'][ci],
              'specialist':owners.classify_entry_specialist_feature(fields['signal'][si])} for si,ci in zip(
        provider.profile['temporal_alias_signal_indices'],provider.profile['temporal_alias_ctx_cont_indices'])]
    surfaces={name:norm.fit_surface_normalization(part,surface=name,field_names=fields[name],
        semantic_categorical_domains=(norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if name=='signal' else {}))
        for name,part in parts.items()}
    surfaces['ctx_cont']=norm.share_temporal_alias_stats_from_signal(surfaces['ctx_cont'],surfaces['signal'],
        temporal_aliases=aliases,ctx_cont_values=parts['ctx_cont'])
    tf_population={}
    for tf in TIMEFRAMES:
        ids=_window_union(provider.tf_tails[tf][rows],WINDOWS[tf],len(provider.tf_times[tf]))
        available=provider.tf_times[tf][ids]+int(htf.MULTI_TF_SHIFT[tf].value)
        _need(np.all(available<cutoff_ns),'FIT_TF_CUTOFF:'+tf)
        key='mtf_'+tf.lower();part=norm.MatrixPopulationPart(provider.tf_values[tf],row_indices=ids,source=key)
        surfaces[key]=norm.fit_surface_normalization(part,surface=key,field_names=fields['mtf'],
            semantic_categorical_domains=norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS)
        tf_population[tf]={'rows':len(ids),'source_rows_sha256':_array_sha(ids,'<i8'),
                           'available_ns_sha256':_array_sha(available,'<i8')}
    cat=norm.fit_ctx_cat_contract(norm.MatrixPopulationPart(raw['ctx_cat'],row_indices=rows,source='native_M5_categories'),
                                   field_names=fields['ctx_cat'])
    population={'cutoff_ns':cutoff_ns,'first_issue_ns':int(provider.issue_ns[rows[0]]),
        'last_issue_ns':int(provider.issue_ns[rows[-1]]),'issue_rows':len(rows),
        'selected_issue_ns_sha256':_array_sha(provider.issue_ns[rows],'<i8'),
        'selected_source_rows_sha256':_array_sha(rows,'<i8'),'local_rows':len(local),'context_rows':len(rows),
        'per_tf':tf_population}
    return norm.build_native_m5_input_normalization_contract(
        native_m5_manifest=raw['source_binding']['native_m5_manifest'],
        feature_provider_contract_sha256=raw['provider_contract']['contract_sha256'],
        source_axes_sha256=provider.source_axes_sha256,fit_population=population,surfaces=surfaces,
        ctx_cat=cat,temporal_aliases=aliases)
