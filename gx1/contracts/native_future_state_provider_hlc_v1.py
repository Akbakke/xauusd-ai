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
