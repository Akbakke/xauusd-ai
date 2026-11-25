"""Explicit TRAIN D1 provider-HLC observations; legacy clock facts stay unchanged.

This owner does no IO. The producer authenticates and loads the referenced files;
this module authenticates the exact provider bytes again before projecting them.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import PurePosixPath
import numpy as np
from gx1.contracts.market_state_native_m5_targets_v1 import (
    TIMEFRAMES,fully_future_state_contract,require_fully_future_state_contract,
)

SCHEMA='gx1_native_m5_fully_future_state_dataset_v2_provider_hlc'
CERTIFICATE_SCHEMA='gx1_native_D1_provider_hlc_certificate_v1'
EROOT='/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/'
DESIGN={'path':EROOT+'D1_PROVIDER_HLC_DESIGN001.json','sha256':'cd97abfec7c512364c21c9608c41352ee70dac9593e1dfd28a5527713734f172'}
SOURCE_BINDINGS={
 'base_dataset':{'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_QUALIFICATION_20261121_001/FUTURE_STATE_DATASET001.json','sha256':'109a2738abf020a528765bac95b7cf73e145a88644dda58a0bc8f95c252ffe30'},
 'base_targets':{'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_QUALIFICATION_20261121_001/FUTURE_STATE_TARGETS001.npz','sha256':'d21b7edd8fbce2afcaecbc7d16ea764d259cf4ed1db219ebb661857acb32f851'},
 'native_manifest':{'path':'/home/andre2/GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927/MANIFEST.json','sha256':'ca0012e5e30f7808876d666fe4d32e3fb873fddb6fc6178bd47b97eb89194cca'},
 'reference_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_OANDA_D1_REFERENCE_20261123_001/RESULT.json','sha256':'ac590829996631664b7b67d5a36b2d9b417046e9dab2ec6feee38918e7693af9'},
 'reference_raw':{'path':'/home/andre2/GX1_RUNS/NATIVE_OANDA_D1_REFERENCE_20261123_001/response-01.canonical.json','sha256':'c7181d30878746126afeb321406f43d2f5dde8c0f208194191c13c5975694cbe'},
 'comparison_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_OANDA_D1_COMPARE_20261123_001/RESULT.json','sha256':'0f7635b911a7e4003e7873b1cc247ec3703b1c34d71ad98c0a903ca427d6a286'},
 'target_join_result':{'path':'/home/andre2/GX1_RUNS/NATIVE_OANDA_D1_TARGET_JOIN_20261123_001/RESULT.json','sha256':'0fd10d8c20ee14ec329a5e105b44109e9ce256e45b8dc386d6bd24367c72e176'},
 'target_window_ledger':{'path':'/home/andre2/GX1_RUNS/NATIVE_OANDA_D1_TARGET_JOIN_20261123_001/D1_WINDOW_LEDGER.json','sha256':'8aef2a2b8264a77e783ac324df2559b2315a7f5d1e77ae7213e1ca41bdea9b31'},
}
SOURCE_IDENTITY={'environment':'practice','instrument':'XAU_USD','granularity':'D','price':'MBA','dailyAlignment':22,'alignmentTimezone':'UTC'}
REQUEST={'from':'2020-12-31T22:00:00Z','to':'2025-12-31T22:00:00Z','granularity':'D','price':'MBA','dailyAlignment':22,'alignmentTimezone':'UTC'}
TRAIN_ISSUE_SHA256='c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea'
TRAIN_UPPER_NS=1767225600000000000
PROVIDER_ROWS=1290
DAY_NS=86400000000000;D1_ORIGIN_NS=79200000000000;D1_INDEX=TIMEFRAMES.index('D1')


def _need(ok,reason):
    if not bool(ok):raise ValueError('NATIVE_PROVIDER_HLC_'+reason)


def _json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _header(value):return {'shape':list(value.shape),'dtype':value.dtype.str,'sha256':_array_sha(value)}


def provider_hlc_state_contract():
    old=fully_future_state_contract();value={k:v for k,v in old.items()if k!='contract_sha256'}
    value.update(schema_version=SCHEMA,base_state_contract_sha256=old['contract_sha256'],
        qualification='D1_provider_relative_HLC_observation_only_requires_separate_generation_and_initial_baseline',
        observation_fields=old['observation_fields']+['provider_hlc_observed','state_hlc_observed'],
        provider_hlc_certificate_schema=CERTIFICATE_SCHEMA,provider_hlc_design=DESIGN.copy(),
        provider_hlc_source_binding_sha256=_json_sha(SOURCE_BINDINGS),
        provider_hlc_scope='TRAIN_mature_D1_UTC22_exact_parent_HLC_only_no_other_TF_promotion',
        state_hlc_observed='state_window_complete_OR_authenticated_provider_hlc_observed',
        activity_known='mature_and_observed_HLC_or_explicit_partial_positive_range_certificate',
        zero_range='observed_HLC_exact_zero_is_FLAT_even_zero_or_missing_ATR_no_epsilon',
        partial_range_active='unchanged_ACTIVE_and_not_state_window_complete_M5_clock_fact_not_HLC_unknown',
        side_certificate='observed_HLC_unique_material_endpoint_excursion_C_minus_L_vs_H_minus_C_same_scale',
        side_known='requires_known_ACTIVE_observed_HLC_and_exclusive_endpoint_excursion_certificate',
        provider_close='last_reported_provider_bar_close_no_nominal_end_second_or_publication_time_assertion',
        provider_hlc_limits='no_allquotehistory_calendar_firstvintage_firsthitorder_inputchange_or_learning_claim')
    return {**value,'contract_sha256':_json_sha(value)}


def require_provider_hlc_state_contract(value):
    expected=provider_hlc_state_contract();_need(type(value)is dict and value==expected,'CONTRACT');return expected


def require_future_state_observation_contract(value):
    if type(value)is dict and value.get('schema_version')==ALL_TF_SCHEMA:return require_all_tf_provider_hlc_state_contract(value)
    if type(value)is dict and value.get('schema_version')==SCHEMA:return require_provider_hlc_state_contract(value)
    return require_fully_future_state_contract(value)


def require_provider_hlc_certificate_ref(value):
    spec=provider_hlc_state_contract();keys={'path','sha256','target_contract_sha256','source_binding_sha256'}
    _need(type(value)is dict and set(value)==keys,'CERTIFICATE_REF_KEYS');path=value['path']
    _need(type(path)is str and PurePosixPath(path).is_absolute()and str(PurePosixPath(path))==path and '..'not in PurePosixPath(path).parts and PurePosixPath(path).suffix=='.json','CERTIFICATE_REF_PATH')
    _need(all(type(value[k])is str and re.fullmatch('[0-9a-f]{64}',value[k])for k in keys-{'path'})and value['target_contract_sha256']==spec['contract_sha256']
        and value['source_binding_sha256']==spec['provider_hlc_source_binding_sha256'],'CERTIFICATE_REF_BINDING');return dict(value)


def certificate_metadata_bytes(metadata):return _json_bytes(metadata)


def _provider_arrays(raw_response):
    _need(type(raw_response)is bytes and 0<len(raw_response)<=1048576 and hashlib.sha256(raw_response).hexdigest()==SOURCE_BINDINGS['reference_raw']['sha256'],'SOURCE_RESPONSE_SHA')
    try:value=json.loads(raw_response)
    except (ValueError,UnicodeError)as error:raise ValueError('NATIVE_PROVIDER_HLC_RESPONSE_JSON')from error
    _need(type(value)is dict and value.get('instrument')=='XAU_USD'and value.get('granularity')=='D'and type(value.get('candles'))is list and len(value['candles'])==PROVIDER_ROWS,'PROVIDER_RESPONSE_IDENTITY')
    starts=[];hlc=[]
    for candle in value['candles']:
        _need(type(candle)is dict and candle.get('complete')is True and type(candle.get('time'))is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T22:00:00(?:\.0{1,9})?Z',candle['time']),'PROVIDER_CANDLE_TIME_COMPLETE')
        try:start=int(np.datetime64(candle['time'][:-1],'ns').astype(np.int64))
        except (ValueError,OverflowError)as error:raise ValueError('NATIVE_PROVIDER_HLC_PROVIDER_TIME')from error
        _need(type(candle.get('volume'))is int and candle['volume']>=0,'PROVIDER_TICKVOLUME')
        for component in ('mid','bid','ask'):
            prices=candle.get(component);_need(type(prices)is dict and set(prices)=={'o','h','l','c'},'PROVIDER_MBA_FIELDS')
            try:price=np.asarray([float(prices[k])for k in ('o','h','l','c')],dtype=np.float64)
            except (ValueError,TypeError,OverflowError)as error:raise ValueError('NATIVE_PROVIDER_HLC_PROVIDER_PRICES')from error
            _need(all(type(prices[k])is str for k in prices)and np.isfinite(price).all()and np.all(price>0)and price[2]<=min(price[0],price[3])<=max(price[0],price[3])<=price[1],'PROVIDER_FINITE_OHLC')
            if component=='mid':hlc.append(price[[1,2,3]])
        starts.append(start)
    starts=np.asarray(starts,dtype=np.int64);ends=starts+DAY_NS;hlc=np.asarray(hlc,dtype=np.float64)
    lower=int(np.datetime64(REQUEST['from'][:-1],'ns').astype(np.int64));upper=int(np.datetime64(REQUEST['to'][:-1],'ns').astype(np.int64))
    _need(len(starts)>0 and np.all(np.diff(starts)>0)and np.all((starts-D1_ORIGIN_NS)%DAY_NS==0)and starts[0]>=lower and ends[-1]<=upper and np.all(ends<TRAIN_UPPER_NS),'PROVIDER_AXIS_DUPLICATE_RANGE_OR_FOLD')
    return {'target_start_ns':starts,'validity_ns':ends,'hlc':hlc}


def make_provider_hlc_certificate(raw_response,*,certificate_path):
    arrays=_provider_arrays(raw_response);spec=provider_hlc_state_contract()
    metadata={'schema_version':CERTIFICATE_SCHEMA,'basis':'PROVIDER_HLC','design':json.loads(json.dumps(DESIGN)),
        'source_bindings':json.loads(json.dumps(SOURCE_BINDINGS)),'source_identity':dict(SOURCE_IDENTITY),'request':dict(REQUEST),
        'base_state_contract_sha256':fully_future_state_contract()['contract_sha256'],'target_contract_sha256':spec['contract_sha256'],
        'scope':'original_whole_TRAIN_D1_only','issue_ns_sha256':TRAIN_ISSUE_SHA256,'fold_upper_ns':TRAIN_UPPER_NS,
        'array_headers':{k:_header(v)for k,v in arrays.items()},'no_calendar_or_firstvintage_or_publication_time_claim':True}
    ref={'path':certificate_path,'sha256':hashlib.sha256(certificate_metadata_bytes(metadata)).hexdigest(),
        'target_contract_sha256':spec['contract_sha256'],'source_binding_sha256':spec['provider_hlc_source_binding_sha256']}
    require_provider_hlc_certificate_ref(ref);return {'metadata':metadata,'raw_response':raw_response,'ref':ref}


def authenticate_provider_hlc_certificate_ref(ref,*,metadata_bytes,raw_response_bytes):
    """Authenticate actual metadata and provider bytes; no IO or caller flags."""
    ref=require_provider_hlc_certificate_ref(ref)
    expected=make_provider_hlc_certificate(raw_response_bytes,certificate_path=ref['path'])
    _need(type(metadata_bytes)is bytes and metadata_bytes==certificate_metadata_bytes(expected['metadata'])and ref==expected['ref'],'CERTIFICATE_AUTHENTICATION')
    return expected


def apply_provider_hlc_observation(parent,*,issue_ns,fold_upper_ns,certificate=None):
    _need(type(parent)is dict,'PARENT_PAYLOAD');require_fully_future_state_contract(parent['future_state_contract'])
    if certificate is None:return parent
    _need(type(certificate)is dict and set(certificate)=={'metadata','raw_response','ref'},'CERTIFICATE_BUNDLE')
    ref=require_provider_hlc_certificate_ref(certificate['ref']);expected=make_provider_hlc_certificate(certificate['raw_response'],certificate_path=ref['path'])
    _need(certificate['metadata']==expected['metadata']and ref==expected['ref'],'CERTIFICATE_CONTENT_BINDING');arrays=_provider_arrays(certificate['raw_response'])
    _need(type(fold_upper_ns)is int and fold_upper_ns==TRAIN_UPPER_NS and type(issue_ns)is np.ndarray and issue_ns.dtype==np.int64 and issue_ns.ndim==1 and len(issue_ns)>0
        and np.all(np.diff(issue_ns)>0)and np.all(issue_ns<fold_upper_ns)and _array_sha(issue_ns)==TRAIN_ISSUE_SHA256,'TRAIN_ISSUE_FOLD_BINDING')
    if 'issue_ns'in parent:_need(np.array_equal(parent['issue_ns'],issue_ns),'PARENT_ISSUE_AXIS')
    shape=(len(issue_ns),len(TIMEFRAMES))
    for key in ('target_present','mature','state_window_complete','nominal_complete','partial_range_active','activity','activity_known','side_known','side_up'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.bool_ and parent[key].shape==shape,'PARENT_BOOL:'+key)
    for key in ('target_start_ns','validity_ns','atr_asof_available_ns'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.int64 and parent[key].shape==shape,'PARENT_TIME:'+key)
    for key in ('target_range','atr14_asof','state_activity_scale'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.float64 and parent[key].shape==shape,'PARENT_VALUE:'+key)
    _need(type(parent.get('target_ohlc'))is np.ndarray and parent['target_ohlc'].dtype==np.float64 and parent['target_ohlc'].shape==shape+(4,),'PARENT_OHLC')
    present=parent['target_present'];mature=parent['mature'];starts=parent['target_start_ns'];ends=parent['validity_ns'];whole=parent['state_window_complete'];d=D1_INDEX
    _need(np.array_equal(mature,present&(ends<fold_upper_ns))and not np.any(whole&~mature),'PARENT_MATURITY')
    ready=mature[:,d];_need(ready.any(),'D1_MATURE_SUPPORT');s=starts[ready,d];e=ends[ready,d]
    _need(np.all(s>=issue_ns[ready])and np.all((s-D1_ORIGIN_NS)%DAY_NS==0)and np.all(e-s==DAY_NS)and np.all(e<fold_upper_ns),'D1_FUTURE_PHASE_END')
    at=np.searchsorted(arrays['target_start_ns'],s);_need(np.all(at<len(arrays['target_start_ns']))and np.array_equal(arrays['target_start_ns'][at],s)and np.array_equal(arrays['validity_ns'][at],e),'D1_PROVIDER_MAPPING')
    target=parent['target_ohlc'][ready,d];hlc=target[:,[1,2,3]]
    _need(np.array_equal(hlc,arrays['hlc'][at])and np.array_equal(parent['target_range'][ready,d],target[:,1]-target[:,2]),'D1_POINTWISE_PARENT_HLC')
    atr=parent['atr14_asof'];scale=np.where(np.isfinite(atr)&(atr>=0),0.5*atr,np.nan)
    _need(np.array_equal(scale,parent['state_activity_scale'],equal_nan=True)and np.all(parent['atr_asof_available_ns'][np.isfinite(atr)]<=np.broadcast_to(issue_ns[:,None],shape)[np.isfinite(atr)]),'PARENT_CAUSAL_ATR_SCALE')
    active=mature&np.isfinite(scale)&(parent['target_range']>0)&(parent['target_range']>=scale)
    _need(np.array_equal(active,parent['activity'])and np.array_equal(parent['partial_range_active'],active&~whole),'UNCHANGED_ACTIVITY_AND_CLOCK_PARTIAL')
    provider=np.zeros(shape,dtype=np.bool_);provider[ready,d]=True;observed=whole|provider
    output=dict(parent);output['provider_hlc_observed']=provider;output['state_hlc_observed']=observed;output['target_observation_certificate_ref']=ref
    known=parent['activity_known'].copy();side=parent['side_known'].copy();up=parent['side_up'].copy()
    flat=observed[:,d]&((parent['target_range'][:,d]==0)|(np.isfinite(scale[:,d])&(parent['target_range'][:,d]<scale[:,d])))
    close=parent['target_ohlc'][:,d,3];low=parent['target_ohlc'][:,d,2];high=parent['target_ohlc'][:,d,1]
    up_material=np.where(scale[:,d]>0,close-low>=scale[:,d],close-low>0);down_material=np.where(scale[:,d]>0,high-close>=scale[:,d],high-close>0)
    known[:,d]=active[:,d]|flat;side[:,d]=observed[:,d]&active[:,d]&(up_material^down_material);up[:,d]=side[:,d]&up_material
    output.update(activity_known=known,side_known=side,side_up=up,future_state_contract=provider_hlc_state_contract());return output


# Explicit successor: existing V2 identities and projection above remain intact.
ALL_TF_SCHEMA='gx1_native_m5_fully_future_state_dataset_v3_all_tf_provider_hlc'
ALL_TF_CERTIFICATE_SCHEMA='gx1_native_all_TF_provider_hlc_certificate_v1'
ALL_TF_SOURCE_BINDINGS={
    **{k:dict(SOURCE_BINDINGS[k]) for k in ('base_dataset','base_targets','native_manifest')},
    'provider_manifest':{'path':EROOT+'ALL_TF_BAR_COMPARE_MANIFEST_20261125_002.json','sha256':'c90f2cb0cc48935b47600e3b2a818d9db8aefc26bbd568f47e7eb00a87fd9adb'},
    'restoration_result':{'path':'/home/andre2/GX1_RUNS/RESTORED_NATIVE_BAR_OBSERVATIONS_20261125_001/RESULT.json','sha256':'f66327bb2a26aec5e35fc0343be416392e7bf52244c0e4dd298adc1aebac5e21'},
    'm5_raw_integrity_acceptance':{'path':EROOT+'FULL_SOURCE_INTEGRITY_ACCEPTANCE001.json','sha256':'a2688d08e18e8f2fba9b047ff4c4ebb30000efc0c865df4e5e217392472a54ce'},
}
ALL_TF_TRAIN_LOWER_NS=int(np.datetime64('2021-01-01','ns').astype(np.int64))
ALL_TF_EVIDENCE_KEYS=('provider_manifest','restoration_result','m5_raw_integrity_acceptance')


def all_tf_provider_hlc_state_contract():
    value={k:v for k,v in provider_hlc_state_contract().items() if k not in ('contract_sha256','provider_hlc_design')}
    value.update(schema_version=ALL_TF_SCHEMA,
        qualification='all_five_TF_observed_HLC_only_new_generation_and_fresh_initial_required',
        provider_hlc_certificate_schema=ALL_TF_CERTIFICATE_SCHEMA,
        provider_hlc_source_binding_sha256=_json_sha(ALL_TF_SOURCE_BINDINGS),
        provider_hlc_reference=ALL_TF_SOURCE_BINDINGS['restoration_result'].copy(),
        provider_hlc_scope='original_whole_TRAIN_M15_H1_H4_D1_exact_parent_HLC_M5_native_complete_only',
        M5_observation_basis='unchanged_native_mature_state_window_complete_and_reused_raw_proof_no_fresh_provider_flag',
        provider_hlc_limits='no_input_restoration_calendar_tradeability_firstvintage_firsthitorder_or_learning_claim')
    return {**value,'contract_sha256':_json_sha(value)}


def require_all_tf_provider_hlc_state_contract(value):
    expected=all_tf_provider_hlc_state_contract();_need(type(value)is dict and value==expected,'ALL_TF_CONTRACT');return expected


def require_all_tf_provider_hlc_certificate_ref(value):
    spec=all_tf_provider_hlc_state_contract();keys={'path','sha256','target_contract_sha256','source_binding_sha256'}
    _need(type(value)is dict and set(value)==keys,'ALL_TF_CERTIFICATE_REF_KEYS');path=value['path']
    _need(type(path)is str and PurePosixPath(path).is_absolute() and str(PurePosixPath(path))==path and '..'not in PurePosixPath(path).parts and PurePosixPath(path).suffix=='.json','ALL_TF_CERTIFICATE_REF_PATH')
    _need(all(type(value[k])is str and re.fullmatch('[0-9a-f]{64}',value[k]) for k in keys-{'path'}) and value['target_contract_sha256']==spec['contract_sha256'] and value['source_binding_sha256']==spec['provider_hlc_source_binding_sha256'],'ALL_TF_CERTIFICATE_REF_BINDING')
    return dict(value)


def _all_tf_time_ns(value):
    _need(type(value)is str and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:00(?:\.0{1,9})?(?:Z|\+00:00)',value),'ALL_TF_UTC_MINUTE')
    text=value[:-1] if value.endswith('Z') else value[:-6]
    try:return int(np.datetime64(text,'ns').astype(np.int64))
    except (ValueError,OverflowError)as error:raise ValueError('NATIVE_PROVIDER_HLC_ALL_TF_TIME')from error


def _all_tf_provider_arrays(evidence_bytes,raw_responses):
    _need(type(evidence_bytes)is dict and set(evidence_bytes)==set(ALL_TF_EVIDENCE_KEYS),'ALL_TF_EVIDENCE_KEYS')
    evidence={}
    for key in ALL_TF_EVIDENCE_KEYS:
        raw=evidence_bytes[key]
        _need(type(raw)is bytes and hashlib.sha256(raw).hexdigest()==ALL_TF_SOURCE_BINDINGS[key]['sha256'],'ALL_TF_EVIDENCE_SHA:'+key)
        try:evidence[key]=json.loads(raw)
        except (ValueError,UnicodeError)as error:raise ValueError('NATIVE_PROVIDER_HLC_ALL_TF_EVIDENCE_JSON')from error
    manifest=evidence['provider_manifest'];result=evidence['restoration_result'];proof=evidence['m5_raw_integrity_acceptance']
    _need(manifest['schema_version']=='gx1_all_tf_bar_compare_v1' and manifest['m5_raw_integrity_acceptance']==ALL_TF_SOURCE_BINDINGS['m5_raw_integrity_acceptance'] and manifest['seal_ns']==int(np.datetime64('2026-07-01','ns').astype(np.int64)),'ALL_TF_REFERENCE_MANIFEST')
    _need(result['status']=='COMPLETE_RESTORED_PROVIDER_OBSERVATION_PARITY' and result['all_higher_TF_13_fields_exact']is True and result['current_native_inputs_changed']is False and result['targets_prior_baseline_changed']is False and result['TEST_used']is False,'ALL_TF_RESTORATION_PROOF')
    _need(proof['status']=='PASS_RAW_MBA_CANONICAL_REPRODUCTION_WITH_DECLARED_FILTER' and proof['measured']['active_input_rows_strictly_before_availability_cutoff']>0,'ALL_TF_M5_REUSED_PROOF')
    responses=manifest['provider_responses'];_need(type(responses)is dict and set(responses)==set(TIMEFRAMES) and responses['M5']==[],'ALL_TF_RESPONSE_SURFACE')
    bindings={}
    for tf in TIMEFRAMES[1:]:
        for entry in responses[tf]:
            binding=entry['response'];_need(type(binding)is dict and set(binding)=={'path','sha256'} and binding['path']not in bindings,'ALL_TF_RESPONSE_REF')
            path=binding['path'];_need(type(path)is str and PurePosixPath(path).is_absolute() and str(PurePosixPath(path))==path and '..'not in PurePosixPath(path).parts,'ALL_TF_RESPONSE_PATH');bindings[path]=binding['sha256']
    _need(type(raw_responses)is dict and set(raw_responses)==set(bindings),'ALL_TF_RAW_RESPONSE_INVENTORY')
    for path,sha in bindings.items():
        raw=raw_responses[path];_need(type(raw)is bytes and hashlib.sha256(raw).hexdigest()==sha,'ALL_TF_RAW_RESPONSE_SHA:'+path)
    spec=all_tf_provider_hlc_state_contract();arrays={}
    for tf in TIMEFRAMES[1:]:
        duration=spec['tf_duration_ns'][tf];origin=spec['tf_origin_ns'][tf];previous_end=None;previous_start=None;starts=[];hlc=[];observed=0
        _need(type(responses[tf])is list and len(responses[tf])>0,'ALL_TF_RESPONSE_LIST:'+tf)
        for entry in responses[tf]:
            request=entry['request'];lower=_all_tf_time_ns(request['from']);upper=_all_tf_time_ns(request['to'])
            _need(lower<upper<=manifest['seal_ns'] and (previous_end is None or lower==previous_end) and request['price']=='MBA' and request['granularity']==('D' if tf=='D1' else tf) and request['dailyAlignment']==22 and request['alignmentTimezone']=='UTC' and request['smooth']is False and request['includeFirst']is True,'ALL_TF_REQUEST:'+tf);previous_end=upper
            try:response=json.loads(raw_responses[entry['response']['path']])
            except (ValueError,UnicodeError)as error:raise ValueError('NATIVE_PROVIDER_HLC_ALL_TF_RESPONSE_JSON')from error
            _need(type(response)is dict and response.get('instrument')=='XAU_USD' and response.get('granularity')==request['granularity'] and type(response.get('candles'))is list,'ALL_TF_PROVIDER_IDENTITY:'+tf)
            for candle in response['candles']:
                _need(type(candle)is dict and candle.get('complete')is True,'ALL_TF_PROVIDER_COMPLETE:'+tf);start=_all_tf_time_ns(candle['time']);end=start+duration
                _need(lower<=start<upper and end<manifest['seal_ns'] and (previous_start is None or start>previous_start) and (start-origin)%duration==0,'ALL_TF_PROVIDER_CLOCK:'+tf);previous_start=start;observed+=1
                if start<ALL_TF_TRAIN_LOWER_NS or end>=TRAIN_UPPER_NS:continue
                _need(type(candle.get('volume'))is int and candle['volume']>=0,'ALL_TF_PROVIDER_VOLUME')
                for component in ('mid','bid','ask'):
                    prices=candle.get(component);_need(type(prices)is dict and set(prices)=={'o','h','l','c'} and all(type(prices[k])is str for k in prices),'ALL_TF_PROVIDER_MBA')
                    try:price=np.asarray([float(prices[k])for k in ('o','h','l','c')],dtype=np.float64)
                    except (ValueError,TypeError,OverflowError)as error:raise ValueError('NATIVE_PROVIDER_HLC_ALL_TF_PRICE')from error
                    _need(np.isfinite(price).all() and np.all(price>0) and price[2]<=min(price[0],price[3])<=max(price[0],price[3])<=price[1],'ALL_TF_PROVIDER_FINITE_OHLC')
                    if component=='mid':hlc.append(price[[1,2,3]])
                starts.append(start)
        after=result['timeframes'][tf]['after']
        _need(after['all_13_fields_exact']is True and after['missing_native']==after['extra_native']==after['mismatching_cells']==0 and observed==after['provider_bars'],'ALL_TF_ACCEPTED_PARITY_POPULATION:'+tf)
        axis=np.asarray(starts,dtype=np.int64);values=np.asarray(hlc,dtype=np.float64)
        _need(len(axis)>0 and values.shape==(len(axis),3) and np.all(np.diff(axis)>0),'ALL_TF_TRAIN_AXIS:'+tf)
        arrays[tf]={'target_start_ns':axis,'validity_ns':axis+duration,'hlc':values}
    return arrays


def _build_all_tf_provider_hlc_certificate(evidence_bytes,raw_responses,certificate_path):
    arrays=_all_tf_provider_arrays(evidence_bytes,raw_responses);spec=all_tf_provider_hlc_state_contract()
    headers={'M5':{'basis':'unchanged_native_state_window_complete','reused_proof':ALL_TF_SOURCE_BINDINGS['m5_raw_integrity_acceptance'].copy()}}
    headers.update({tf:{key:_header(value)for key,value in data.items()} for tf,data in arrays.items()})
    metadata={'schema_version':ALL_TF_CERTIFICATE_SCHEMA,'basis':'ALL_TF_OBSERVED_HLC_M5_NATIVE_HIGHER_PROVIDER','source_bindings':json.loads(json.dumps(ALL_TF_SOURCE_BINDINGS)),
        'target_contract_sha256':spec['contract_sha256'],'base_state_contract_sha256':fully_future_state_contract()['contract_sha256'],
        'scope':'original_whole_TRAIN_all_five_TF_observations_only','timeframes':list(TIMEFRAMES),'issue_ns_sha256':TRAIN_ISSUE_SHA256,'fold_upper_ns':TRAIN_UPPER_NS,
        'array_headers_by_tf':headers,'M5_fresh_provider_flag':False,'no_calendar_tradeability_firstvintage_or_publication_time_claim':True}
    ref={'path':certificate_path,'sha256':hashlib.sha256(certificate_metadata_bytes(metadata)).hexdigest(),'target_contract_sha256':spec['contract_sha256'],'source_binding_sha256':spec['provider_hlc_source_binding_sha256']}
    require_all_tf_provider_hlc_certificate_ref(ref)
    return {'metadata':metadata,'evidence_bytes':evidence_bytes,'raw_responses':raw_responses,'ref':ref},arrays


def make_all_tf_provider_hlc_certificate(evidence_bytes,raw_responses,*,certificate_path):
    return _build_all_tf_provider_hlc_certificate(evidence_bytes,raw_responses,certificate_path)[0]


def authenticate_all_tf_provider_hlc_certificate_ref(ref,*,metadata_bytes,evidence_bytes,raw_responses):
    ref=require_all_tf_provider_hlc_certificate_ref(ref);expected,_=_build_all_tf_provider_hlc_certificate(evidence_bytes,raw_responses,ref['path'])
    _need(type(metadata_bytes)is bytes and metadata_bytes==certificate_metadata_bytes(expected['metadata']) and ref==expected['ref'],'ALL_TF_CERTIFICATE_AUTHENTICATION');return expected


def apply_all_tf_provider_hlc_observation(parent,*,issue_ns,fold_upper_ns,certificate):
    _need(type(parent)is dict,'ALL_TF_PARENT');require_fully_future_state_contract(parent['future_state_contract'])
    _need(type(certificate)is dict and set(certificate)=={'metadata','evidence_bytes','raw_responses','ref'},'ALL_TF_CERTIFICATE_BUNDLE')
    ref=require_all_tf_provider_hlc_certificate_ref(certificate['ref']);expected,arrays=_build_all_tf_provider_hlc_certificate(certificate['evidence_bytes'],certificate['raw_responses'],ref['path'])
    _need(certificate['metadata']==expected['metadata'] and ref==expected['ref'],'ALL_TF_CERTIFICATE_CONTENT')
    _need(type(fold_upper_ns)is int and fold_upper_ns==TRAIN_UPPER_NS and type(issue_ns)is np.ndarray and issue_ns.dtype==np.int64 and issue_ns.ndim==1 and len(issue_ns)>0 and np.all(np.diff(issue_ns)>0) and np.all(issue_ns<fold_upper_ns) and _array_sha(issue_ns)==TRAIN_ISSUE_SHA256,'ALL_TF_TRAIN_CLOCK')
    _need(np.array_equal(parent.get('issue_ns'),issue_ns),'ALL_TF_PARENT_ISSUES');shape=(len(issue_ns),len(TIMEFRAMES))
    for key in ('target_present','mature','state_window_complete','nominal_complete','partial_range_active','activity','activity_known','side_known','side_up'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.bool_ and parent[key].shape==shape,'ALL_TF_PARENT_BOOL:'+key)
    for key in ('target_start_ns','validity_ns','atr_asof_available_ns'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.int64 and parent[key].shape==shape,'ALL_TF_PARENT_TIME:'+key)
    for key in ('target_range','atr14_asof','state_activity_scale'):
        _need(type(parent.get(key))is np.ndarray and parent[key].dtype==np.float64 and parent[key].shape==shape,'ALL_TF_PARENT_VALUE:'+key)
    _need(type(parent.get('target_ohlc'))is np.ndarray and parent['target_ohlc'].dtype==np.float64 and parent['target_ohlc'].shape==shape+(4,),'ALL_TF_PARENT_OHLC')
    present=parent['target_present'];mature=parent['mature'];whole=parent['state_window_complete'];starts=parent['target_start_ns'];ends=parent['validity_ns'];prices=parent['target_ohlc'];spec=all_tf_provider_hlc_state_contract()
    _need(np.array_equal(mature,present&(ends<fold_upper_ns)) and not np.any(whole&~mature) and np.array_equal(whole[:,0],mature[:,0]),'ALL_TF_MATURE_M5_NATIVE_COMPLETE')
    provider=np.zeros(shape,dtype=np.bool_)
    for j,tf in enumerate(TIMEFRAMES):
        ready=mature[:,j];s=starts[ready,j];e=ends[ready,j];duration=spec['tf_duration_ns'][tf];origin=spec['tf_origin_ns'][tf]
        _need(ready.any() and np.all(s>=issue_ns[ready]) and np.all((s-origin)%duration==0) and np.all(e-s==duration) and np.all(e<fold_upper_ns),'ALL_TF_FUTURE_PHASE_END:'+tf)
        target=prices[ready,j];_need(np.isfinite(target).all() and np.all(target>0) and np.all(target[:,2]<=np.minimum(target[:,0],target[:,3])) and np.all(np.maximum(target[:,0],target[:,3])<=target[:,1]) and np.array_equal(parent['target_range'][ready,j],target[:,1]-target[:,2]),'ALL_TF_PARENT_FINITE_RANGE:'+tf)
        if j==0:continue
        source=arrays[tf];at=np.searchsorted(source['target_start_ns'],s)
        _need(np.all(at<len(source['target_start_ns'])) and np.array_equal(source['target_start_ns'][at],s) and np.array_equal(source['validity_ns'][at],e),'ALL_TF_PROVIDER_MAPPING:'+tf)
        _need(np.array_equal(target[:,[1,2,3]],source['hlc'][at]),'ALL_TF_POINTWISE_PARENT_HLC:'+tf);provider[ready,j]=True
    atr=parent['atr14_asof'];valid=np.isfinite(atr)&(atr>=0);scale=np.where(valid,0.5*atr,np.nan);issue_grid=np.broadcast_to(issue_ns[:,None],shape)
    _need(np.array_equal(scale,parent['state_activity_scale'],equal_nan=True) and np.all(parent['atr_asof_available_ns'][valid]>0) and np.all(parent['atr_asof_available_ns'][valid]<=issue_grid[valid]),'ALL_TF_CAUSAL_ATR_SCALE')
    ranges=parent['target_range'];active=mature&np.isfinite(scale)&(ranges>0)&(ranges>=scale)
    _need(np.array_equal(active,parent['activity']) and np.array_equal(parent['partial_range_active'],active&~whole),'ALL_TF_UNCHANGED_ACTIVITY_CLOCK')
    observed=whole|provider;flat=observed&((ranges==0)|(np.isfinite(scale)&(ranges<scale)))
    close=prices[:,:,3];low=prices[:,:,2];high=prices[:,:,1];up_material=np.where(scale>0,close-low>=scale,close-low>0);down_material=np.where(scale>0,high-close>=scale,high-close>0)
    known=active|flat;side=observed&active&(up_material^down_material);up=side&up_material
    _need(not np.any(parent['activity_known']&~known) and not np.any(parent['side_known']&~side) and np.array_equal(parent['side_up'][parent['side_known']],up[parent['side_known']]),'ALL_TF_PRESERVE_OLD_KNOWN_FACTS')
    return {**parent,'provider_hlc_observed':provider,'state_hlc_observed':observed,'activity_known':known,'side_known':side,'side_up':up,'target_observation_certificate_ref':ref,'future_state_contract':spec}
