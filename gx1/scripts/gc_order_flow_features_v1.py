"""Single-use, capped GC feature build; no labels, fit, TEST or native launch."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gx1.features import gc_order_flow_v1 as features
from gx1.contracts import gc_continuous_source_v1 as source
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.features.htf_features import MULTI_TF_TIMEFRAMES, MULTI_TF_SHIFT, multi_tf_bar_label
from gx1.scripts import gc_source_disposition_v1 as previous
from gx1.scripts.gc_continuous_source_v1 import load_metadata

REPO = previous.REPO
PLAN_ID = 'GC_M5_FEATURE_BUILD_20261017_006'
PLAN_REL = f'configs/research/{PLAN_ID}.json'
ROOT = Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/PLAN_ID
CODE_PATHS = (
 'gx1/features/gc_order_flow_v1.py', 'gx1/scripts/gc_order_flow_features_v1.py',
 'tests/test_gc_order_flow_features_v1.py', 'gx1/features/smc_v1.py', 'gx1/features/htf_features.py',
 'gx1/contracts/gc_continuous_source_v1.py', 'gx1/scripts/gc_continuous_source_v1.py',
 'gx1/contracts/gc_source_disposition_v1.py', 'gx1/scripts/gc_source_disposition_v1.py',
 'gx1/contracts/gc_order_flow_source_v1.py', 'gx1/scripts/research_ta_campaign_v1.py',
 'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
 'gx1/contracts/entry_model_native_bundle_commit_v1.py',
)
WORKER_COUNT = 8


def runtime(binding_path):
    if binding_path != ROOT/'EXECUTION_BINDING.json':
        raise RuntimeError('GC_FEATURE_BINDING_PATH_INVALID')
    binding = previous.read_json(binding_path)
    previous.verify_digest(REPO/PLAN_REL,binding['plan_sha256'],'PLAN')
    plan = previous.read_json(REPO/PLAN_REL)
    previous.verify_digest(REPO/plan['recipe_document']['path'],plan['recipe_document']['sha256'],'RECIPE_DOCUMENT')
    previous.verify_digest(ROOT/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'APPROVAL')
    approval = previous.read_json(ROOT/'OPERATOR_APPROVAL.json')
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY')
    policy = previous.read_json(REPO/'NEXT_RUN_POLICY.json')
    gate = policy['current_work']['gc_m5_feature_build_006']
    if (plan['plan_id'] != PLAN_ID or previous.git_head() != binding['source_head']
            or approval['approved_plan_sha256'] != binding['plan_sha256']
            or approval['single_use'] is not True or approval['max_invocations'] != 1
            or approval['status'] != 'APPROVED_SINGLE_USE_PENDING_EXECUTION'
            or gate['status'] != 'EXECUTION_CLAIMED' or gate['claimed_invocations'] != 1
            or gate['completed_invocations'] != 0 or gate['execution_authorized'] is not False
            or gate['plan_sha256'] != binding['plan_sha256'] or policy['training_enabled'] is not False):
        raise RuntimeError('GC_FEATURE_SINGLE_USE_AUTHORITY_INVALID')
    if set(binding['code_sha256']) != set(CODE_PATHS):
        raise RuntimeError('GC_FEATURE_CODE_INVENTORY_INVALID')
    for path,digest in binding['code_sha256'].items():
        previous.verify_digest(REPO/path,digest,'CODE')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):
        raise RuntimeError('GC_FEATURE_DIRTY_SOURCE')
    if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip() != 'work/gx1-current':
        raise RuntimeError('GC_FEATURE_BRANCH_INVALID')
    proof = require_capped_cpu_audit_execution()
    proof['lock'] = require_capped_lock_ancestry()
    if proof['cpu_affinity'] != list(range(8)):
        raise RuntimeError('GC_FEATURE_AFFINITY_INVALID')
    if (ROOT/'RESULTS').exists() or (ROOT/'.RESULTS.staging').exists():
        raise RuntimeError('GC_FEATURE_OUTPUT_ALREADY_EXISTS')
    previous.write_json(ROOT/'EXECUTION_CLAIM.json',{'status':'CLAIMED','invocation':1,
        'plan_sha256':binding['plan_sha256'],'binding_sha256':previous.sha256_file(binding_path),
        'source_head':binding['source_head'],'claimed_utc':datetime.now(timezone.utc).isoformat()})
    return plan,binding,proof


def load_admission(plan):
    for item in plan['admitted_inputs'].values():
        previous.verify_digest(Path(item['path']),item['sha256'],'ADMITTED_INPUT')
    report = previous.read_json(Path(plan['admitted_inputs']['result']['path']))
    for name in ('volume','volume_profile','footprint','delta','cvd'):
        if report['summary']['indicator_admission'][name]['decision'] != 'ADMIT':
            raise RuntimeError('GC_FEATURE_SOURCE_NOT_ADMITTED')
    table = pq.read_table(plan['admitted_inputs']['masks']['path'])
    width = (timing.WINDOW_END_NS-timing.STUDY_START_NS)//timing.MINUTE_NS
    minutes = timing.STUDY_START_NS+np.arange(width,dtype=np.int64)*timing.MINUTE_NS
    indices = (table['bar_start_ns'].to_numpy()-timing.STUDY_START_NS)//timing.MINUTE_NS
    if np.any(np.diff(indices)<=0) or np.any(indices<0) or np.any(indices>=width):
        raise RuntimeError('GC_FEATURE_MASK_CLOCK_INVALID')
    selected = np.full(width,-1,dtype=np.int16)
    selected[indices] = table['contract_index'].fill_null(-1).to_numpy()
    metrics = {n:np.zeros(width,dtype=np.uint64) for n in source.METRICS}
    for name,a in metrics.items():
        a[indices] = table[name].to_numpy()
    masks = {n:np.zeros(width,dtype=bool) for n in ('volume','volume_profile','footprint','delta','cvd')}
    for name,a in masks.items():
        a[indices] = table[name+'_input_valid'].to_numpy()
    if np.any(minutes[indices]+timing.MINUTE_NS>=timing.WINDOW_END_NS):
        raise RuntimeError('GC_FEATURE_TEST_BOUNDARY_INVALID')
    if (not np.array_equal(masks['delta'],masks['cvd'])
            or not np.array_equal(masks['delta'],masks['footprint'])
            or not np.array_equal(masks['volume'],masks['volume_profile'])):
        raise RuntimeError('GC_FEATURE_MASK_OWNER_PARITY_FAILED')
    schedule = previous.read_json(Path(plan['admitted_inputs']['roll']['path']))
    return minutes,selected,metrics,masks,schedule


def scan_feature_symbol(group, number, selected, metrics, masks, tick_size, directory, reference_bars):
    collector = features.TradeLadderCollector(None if directory is None else directory/(group['raw_symbol']+'.parquet'),
        selected,masks['volume'],number,tick_size,reference_bars=reference_bars)
    try:
        result = source.scan_symbol(group['items'],tick_size,on_batch=collector)
    finally:
        collector.close()
    take = selected==number
    for name in source.METRICS:
        if not np.array_equal(result[name][take],metrics[name][take]):
            raise RuntimeError('GC_FEATURE_ADMITTED_MINUTE_PARITY_FAILED:'+name)
    valid = take & masks['volume']
    if not np.all(collector.ohlc[valid]>0):
        raise RuntimeError('GC_FEATURE_VALID_PRICE_MISSING')
    return {'files':result['files'],'minute_index':np.flatnonzero(valid),
        'ohlc':collector.ohlc[valid],'symbol':group['raw_symbol'],'collected_trades':collector.rows,'reference_trades':collector.reference_trades}


def load_native_coordinates(plan):
    times,coordinates,splits = [],[],[]
    for split,item in plan['native_datasets'].items():
        for key in ('parquet','manifest'):
            previous.verify_digest(Path(item[key]['path']),item[key]['sha256'],'NATIVE_'+key.upper())
        f = pq.ParquetFile(item['parquet']['path'])
        offset = 0
        for batch in f.iter_batches(columns=['time'],batch_size=65536):
            t = batch.column(0).to_numpy().astype('datetime64[ns]').astype(np.int64)
            take = (t>=timing.STUDY_START_NS) & (t+5*timing.MINUTE_NS<timing.WINDOW_END_NS)
            times.append(t[take]); coordinates.append((offset+np.arange(len(t)))[take])
            splits.extend([split]*int(take.sum()))
            offset += len(t)
    t,rows = np.concatenate(times),np.concatenate(coordinates)
    order = np.argsort(t,kind='stable')
    t = t[order]
    if np.any(np.diff(t)<=0) or np.any(t%(5*timing.MINUTE_NS)):
        raise RuntimeError('GC_FEATURE_NATIVE_CLOCK_INVALID')
    return t,rows[order],np.array(splits)[order]


def bind_m5_rows(times, rows, splits, surfaces):
    decision = times+5*timing.MINUTE_NS
    out = {'time':pa.array(times,type=pa.timestamp('ns',tz='UTC')),
           'decision_ns':decision,'native_row_index':rows,'native_split':splits}
    for tf,surface in surfaces.items():
        duration = MULTI_TF_SHIFT[tf].value
        labels = multi_tf_bar_label(pd.DatetimeIndex(decision-duration,tz='UTC'),tf).asi8
        idx = np.searchsorted(surface['bar_start_ns'],labels)
        inside = idx<len(surface['bar_start_ns'])
        clipped = np.minimum(idx,len(surface['bar_start_ns'])-1)
        matched = inside & (surface['bar_start_ns'][clipped]==labels)
        if np.any(surface['available_ns'][clipped][matched]>decision[matched]):
            raise RuntimeError('GC_FEATURE_MTF_FUTURE_INPUT')
        out[tf.lower()+'_bar_start_ns'] = pa.array(labels,mask=~matched)
        for name,values in surface.items():
            if name in ('bar_start_ns','available_ns','contract_index'):
                continue
            a = np.asarray(values)[clipped]
            if a.dtype==bool:
                out[tf.lower()+'_'+name] = matched & a
            else:
                out[tf.lower()+'_'+name] = pa.array(a,mask=~matched | ~np.isfinite(a))
        out[tf.lower()+'_contract_index'] = pa.array(surface['contract_index'][clipped],
            mask=~matched | ~surface['volume_valid'][clipped])
    if 'M5' in surfaces:
        out['gc_m5_abc_input_valid'] = (np.asarray(out['m5_volume_valid']) &
            np.asarray(out['m5_signed_valid']) & np.isfinite(np.asarray(out['m5_spot_basis_bps'])))
    return pa.table(out)



def reference_bar_selection(minutes, selected, masks, symbol_count):
    months = minutes[::5].astype('datetime64[ns]').astype('datetime64[M]')
    choices = selected.reshape(-1,5)
    same = (choices==choices[:,0,None]).all(axis=1)
    result = []
    for number in range(symbol_count):
        bars = set()
        for mask in (masks['volume'],masks['delta']):
            valid = mask.reshape(-1,5).all(axis=1) & same & (choices[:,0]==number)
            for month in np.unique(months):
                found = np.flatnonzero(valid & (months==month))
                if len(found):
                    bars.add(int(found[0]))
        result.append(sorted(bars))
    return result


def verify_real_references(references, surface, tick_size):
    """Independent raw-Trade Python reduction versus the sparse Arrow path."""
    evidence = []
    scale = tick_size/features.PRICE_SCALE
    for bar,raw in sorted(references.items()):
        histogram = {}
        for price,size,side in raw:
            histogram.setdefault(price,{'B':0,'A':0,'N':0})[side] += size
        prices = sorted(histogram)
        volume_by_price = {p:sum(histogram[p].values()) for p in prices}
        volume = sum(volume_by_price.values())
        cumulative = 0
        quantiles = {}
        for price in prices:
            cumulative += volume_by_price[price]
            for name,num,den in (('q25_usd',1,4),('median_usd',1,2),('q75_usd',3,4)):
                if name not in quantiles and cumulative*den >= volume*num:
                    quantiles[name] = price*scale
        expected = {'volume':volume,'open_usd':raw[0][0]*scale,'close_usd':raw[-1][0]*scale,
            'high_usd':max(prices)*scale,'low_usd':min(prices)*scale,
            'vwap_usd':sum(p*volume_by_price[p] for p in prices)/volume*scale,
            'poc_usd':min(p for p in prices if volume_by_price[p]==max(volume_by_price.values()))*scale,
            'price_levels':len(prices),**quantiles}
        if surface['signed_valid'][bar]:
            delta = [histogram[p]['B']-histogram[p]['A'] for p in prices]
            expected.update(delta_contracts=sum(delta),max_level_delta_contracts=max(delta),
                            min_level_delta_contracts=min(delta))
        for name,value in expected.items():
            if surface[name][bar] != value:
                raise RuntimeError('GC_FEATURE_RAW_REFERENCE_PARITY_FAILED:'+name)
        evidence.append({'m5_bar_start_ns':int(surface['bar_start_ns'][bar]),
            'raw_trade_records':len(raw),'checked_fields':list(expected),'parity':'EXACT'})
    if len(evidence)==0:
        raise RuntimeError('GC_FEATURE_NO_REAL_REFERENCE_BARS')
    return {'raw_reference_bars':len(evidence),'raw_references':evidence}


def verify_real_prefixes(surfaces, minutes, selected, masks, metrics, tick_size):
    checks = []
    from gx1.features.smc_v1 import replay_volume_order_block_proxies_v1
    for tf,surface in surfaces.items():
        valid = surface['volume_valid']
        duration = MULTI_TF_SHIFT[tf].value
        split = np.r_[True, ~valid[:-1] | ~valid[1:] |
            (np.diff(surface['bar_start_ns'])!=duration) | (np.diff(surface['contract_index'])!=0)]
        # Longest real uninterrupted segment, chosen without outcome access.
        starts = np.flatnonzero(valid & split)
        runs = []
        for start in starts:
            end = start+1
            while end<len(valid) and valid[end] and not split[end]:
                end += 1
            runs.append((end-start,int(start),end))
        if not runs:
            checks.append({'timeframe':tf,'status':'NO_VALID_COMPLETE_BARS'})
            continue
        length,start,end = (int(v) for v in max(runs))
        raw = [surface[n][start:end].copy() for n in ('open_usd','high_usd','low_usd','close_usd','volume')]
        cut = max(1,length//2)
        full = replay_volume_order_block_proxies_v1(*raw)
        prefix = replay_volume_order_block_proxies_v1(*[a[:cut] for a in raw])
        changed = [a.copy() for a in raw]
        for a in changed[:4]:
            a[cut:] += tick_size/features.PRICE_SCALE
        future = replay_volume_order_block_proxies_v1(*changed)
        for name in full:
            if not (np.array_equal(full[name][:cut],prefix[name],equal_nan=True) and
                    np.array_equal(full[name][:cut],future[name][:cut],equal_nan=True)):
                raise RuntimeError('GC_FEATURE_REAL_SMC_PREFIX_OR_MUTATION_FAILED')
        checks.append({'timeframe':tf,'real_segment_bars':length,'prefix_bars':cut,
                       'future_suffix_bars':length-cut,'prefix_and_future_mutation':'EXACT'})
    cut = len(minutes)//2
    full = features.relative_cvd(minutes,selected,masks,metrics)
    prefix = features.relative_cvd(minutes[:cut],selected[:cut],
        {k:v[:cut] for k,v in masks.items()},{k:v[:cut] for k,v in metrics.items()})
    for name in full:
        if not np.array_equal(full[name][:cut],prefix[name],equal_nan=True):
            raise RuntimeError('GC_FEATURE_REAL_CVD_PREFIX_FAILED')
    checks.append({'indicator':'relative_cvd','prefix_minutes':cut,'prefix':'EXACT'})
    return checks


def add_basis(plan,surfaces,minutes):
    item = plan['inputs']['oanda_manifest']
    previous.verify_digest(Path(item['path']),item['sha256'],'OANDA_MANIFEST')
    hashes = [(Path(item['path']),item['sha256'])]
    mid = np.full(len(minutes),np.nan)
    for part in item['partitions']:
        path = Path(part['path'])
        previous.verify_digest(path,part['sha256'],'OANDA_PRICE_PARTITION')
        hashes.append((path,part['sha256']))
        for batch in pq.ParquetFile(path).iter_batches(columns=['time','bid_close','ask_close'],batch_size=65536):
            stamp = batch.column(0).to_numpy().astype('datetime64[ns]').astype(np.int64)
            if np.any(stamp>=timing.WINDOW_END_NS):
                raise RuntimeError('GC_FEATURE_OANDA_TEST_BOUNDARY_INVALID')
            bid,ask = (batch.column(i).to_numpy() for i in (1,2))
            take = (stamp>=timing.STUDY_START_NS) & (stamp+timing.MINUTE_NS<timing.WINDOW_END_NS)
            if (not np.isfinite(bid[take]).all() or not np.isfinite(ask[take]).all()
                    or np.any(bid[take]<=0) or np.any(ask[take]<bid[take])):
                raise RuntimeError('GC_FEATURE_OANDA_MID_INVALID')
            idx = (stamp[take]-timing.STUDY_START_NS)//timing.MINUTE_NS
            if np.any(np.isfinite(mid[idx])) or np.any(stamp[take]%timing.MINUTE_NS):
                raise RuntimeError('GC_FEATURE_OANDA_TIME_DUPLICATE_OR_GRID_INVALID')
            mid[idx] = (bid[take]+ask[take])/2
    for surface in surfaces.values():
        end = surface['available_ns']-timing.MINUTE_NS
        idx = (end-timing.STUDY_START_NS)//timing.MINUTE_NS
        take = (idx>=0) & (idx<len(mid)) & surface['volume_valid']
        basis = np.full(len(idx),np.nan)
        basis[take] = 1e4*(surface['close_usd'][take]/mid[idx[take]]-1)
        surface['spot_basis_bps'] = basis
    return hashes


def run(binding_path):
    plan,binding,proof = runtime(binding_path)
    groups,gc_hashes = previous._bound_gc_inputs(plan)
    metadata,metadata_hashes = load_metadata(plan,groups)
    minutes,selected,metrics,masks,schedule = load_admission(plan)
    if [g['raw_symbol'] for g in groups] != schedule['contract_order']:
        raise RuntimeError('GC_FEATURE_CONTRACT_ORDER_CHANGED')
    ticks = {m[2] for m in metadata.values()}
    if len(ticks)!=1:
        raise RuntimeError('GC_FEATURE_NONUNIFORM_TICK_SIZE')
    tick_size = next(iter(ticks))
    stage = ROOT/'.RESULTS.staging'
    stage.mkdir(exist_ok=False)
    # Resume only the bound, completed sparse price-level reductions. OHLC
    # receive order was not persisted by failed 005, so recover that from the
    # original source owner and independently recheck the reused histograms.
    reused = plan['reuse_ladders']
    if set(reused) != {g['raw_symbol'] for g in groups}:
        raise RuntimeError('GC_FEATURE_REUSE_INVENTORY_INVALID')
    for item in reused.values():
        previous.verify_digest(Path(item['path']),item['sha256'],'REUSED_LADDER')
        if Path(item['path']).stat().st_size != item['bytes']:
            raise RuntimeError('GC_FEATURE_REUSED_LADDER_SIZE_INVALID')
    ohlc = np.zeros((len(minutes),4),dtype=np.int64)
    reports = []
    references = {}
    # First complete unsigned/signed M5 per selected contract and month, chosen
    # only from masks, before prices or outcomes are accessed.
    ref_bars = reference_bar_selection(minutes,selected,masks,len(groups))
    with ProcessPoolExecutor(max_workers=WORKER_COUNT,mp_context=mp.get_context('fork')) as pool:
        jobs = [pool.submit(scan_feature_symbol,g,i,selected,metrics,masks,tick_size,None,ref_bars[i])
                for i,g in enumerate(groups)]
        for job in jobs:
            result = job.result()
            ohlc[result.pop('minute_index')] = result.pop('ohlc')
            references.update(result.pop('reference_trades'))
            reports.append(result)
    old = previous.read_json(Path(plan['inputs']['gc_result']['path']))
    counts = {f['path']:(f['counters']['records'],f['counters']['trade_records']) for f in old['files']}
    for report in reports:
        for f in report['files']:
            if counts[f['path']] != (f['records'],f['raw_trade_records']):
                raise RuntimeError('GC_FEATURE_SOURCE_RECORD_RECONCILIATION_FAILED')
    print('GC_FEATURE_RAW_PASS_COMPLETE',flush=True)
    surfaces,maps = {},{}
    for tf in MULTI_TF_TIMEFRAMES:
        surfaces[tf],maps[tf] = features.timeframe_skeleton(minutes,selected,metrics,masks,ohlc,tf,tick_size)
    for g in groups:
        features.consume_ladder(Path(reused[g['raw_symbol']]['path']),surfaces,maps,tick_size)
    cvd = features.relative_cvd(minutes,selected,masks,metrics)
    for tf,surface in surfaces.items():
        if not np.isfinite(surface['vwap_usd'][surface['volume_valid']]).all():
            raise RuntimeError('GC_FEATURE_VALID_PROFILE_MISSING')
        features.add_structure_and_cvd(surface,cvd,maps[tf],minutes)
    real_parity = verify_real_references(references,surfaces['M5'],tick_size)
    real_parity['prefix_checks'] = verify_real_prefixes(surfaces,minutes,selected,masks,metrics,tick_size)
    oanda_hashes = add_basis(plan,surfaces,minutes)
    previous.write_json(stage/'REAL_PARITY.json',real_parity)
    times,rows,splits = load_native_coordinates(plan)
    table = bind_m5_rows(times,rows,splits,surfaces)
    # Whole-population lookup parity: M5 identity/availability and every TF's
    # canonical last-closed label are checked independently of serialization.
    for tf in MULTI_TF_TIMEFRAMES:
        from gx1.features.htf_features import multi_tf_last_closed_label
        probes = np.unique(np.r_[0,len(times)//2,len(times)-1])
        for i in probes:
            expected = multi_tf_last_closed_label(pd.Timestamp(int(times[i]),tz='UTC'),tf).value
            actual = table[tf.lower()+'_bar_start_ns'][int(i)].as_py()
            if actual is not None and actual != expected:
                raise RuntimeError('GC_FEATURE_CANONICAL_MTF_LABEL_PARITY_FAILED')
    pq.write_table(table,stage/'M5_FEATURES.parquet',compression='zstd')
    if not table.equals(pq.read_table(stage/'M5_FEATURES.parquet')):
        raise RuntimeError('GC_FEATURE_PARQUET_PARITY_FAILED')
    # No new data or source mutation can pass an atomic publication.
    previous._rehash_sources(gc_hashes,oanda_hashes)
    for path,digest in metadata_hashes:
        previous.verify_digest(path,digest,'SOURCE_POST_SCAN')
    for item in plan['admitted_inputs'].values():
        previous.verify_digest(Path(item['path']),item['sha256'],'ADMISSION_POST_SCAN')
    for item in reused.values():
        previous.verify_digest(Path(item['path']),item['sha256'],'REUSED_LADDER_POST_SCAN')
    for path,digest in binding['code_sha256'].items():
        previous.verify_digest(REPO/path,digest,'CODE_POST_SCAN')
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY_POST_SCAN')
    previous.verify_digest(REPO/PLAN_REL,binding['plan_sha256'],'PLAN_POST_SCAN')
    previous.verify_digest(REPO/plan['recipe_document']['path'],plan['recipe_document']['sha256'],'RECIPE_POST_SCAN')
    previous.verify_digest(ROOT/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'APPROVAL_POST_SCAN')
    claim = previous.read_json(ROOT/'EXECUTION_CLAIM.json')
    previous.verify_digest(binding_path,claim['binding_sha256'],'BINDING_POST_SCAN')
    for item in plan['native_datasets'].values():
        for key in ('parquet','manifest'):
            previous.verify_digest(Path(item[key]['path']),item[key]['sha256'],'NATIVE_POST_SCAN')
    if previous.git_head()!=binding['source_head']:
        raise RuntimeError('GC_FEATURE_HEAD_CHANGED')
    support = {}
    months = times.astype('datetime64[ns]').astype('datetime64[M]')
    for tf in MULTI_TF_TIMEFRAMES:
        support[tf] = {}
        for name in ('volume_valid','signed_valid','ob_bull_input_valid','ob_bear_input_valid'):
            column = table[tf.lower()+'_'+name]
            values = column.to_numpy(zero_copy_only=False)
            valid = np.isfinite(values) & (values==1)
            labels = np.array(table[tf.lower()+'_bar_start_ns'].to_pylist(),dtype=np.float64)
            support[tf][name] = {'native_m5_rows':int(valid.sum()),
                'unique_closed_tf_bars':len(np.unique(labels[valid])),
                'monthly_native_m5_rows':{str(m):int(np.count_nonzero(valid & (months==m))) for m in np.unique(months)}}
    result = {'status':'FEATURE_BUILD_COMPLETE_NOT_TRAINING_ADMISSION','plan_id':PLAN_ID,
        'plan_sha256':binding['plan_sha256'],'source_head':binding['source_head'],
        'code_sha256':binding['code_sha256'],'native_m5_rows':len(times),'support':support,
        'source_reports':reports,'limits':proof,'real_parity':real_parity,'reused_ladders':reused,
        'feature_columns':table.schema.names,
        'abc_gc_input_valid_rows':int(np.count_nonzero(table['gc_m5_abc_input_valid'].to_numpy())),'test_accessed':False,'outcomes_read':False,
        'fit_started':False,'economics_evaluated':False,'native_train_serve_parity_proven':False,
        'source_complete':False,'new_databento_requests':0,'new_spending':False}
    previous.write_json(stage/'RESULT.json',result)
    artifacts = {}
    for path in sorted(stage.rglob('*')):
        if not path.is_file():
            continue
        with path.open('rb') as f:
            os.fsync(f.fileno())
        artifacts[str(path.relative_to(stage))] = {'sha256':previous.sha256_file(path),'bytes':path.stat().st_size}
    previous.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'invocation':1,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'artifacts':artifacts,
        'test_accessed':False,'fit_started':False,'model_admitted':False})
    if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} != set(artifacts)|{'TERMINAL.json'}:
        raise RuntimeError('GC_FEATURE_PUBLICATION_INVENTORY_MISMATCH')
    for name,entry in artifacts.items():
        previous.verify_digest(stage/name,entry['sha256'],'PUBLISHED_ARTIFACT')
    for path in stage.glob('*.json'):
        previous.read_json(path)
    for directory in (stage,):
        fd = os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    publish_bundle_directory_noreplace(stage,ROOT/'RESULTS')
    print(json.dumps({k:result[k] for k in ('status','native_m5_rows','support')},sort_keys=True),flush=True)


def foundation_worker_v2(group,tick,wanted,candidate,expected,references):
    c=features.L1FeatureCollectorV2(wanted,tick,reference_minutes=references)
    old=source.scan_symbol(group['items'],tick,on_batch=c,extra_columns=source.L1_AUDIT_COLUMNS,on_file_start=c.file_start)
    c.finish()
    for key in source.METRICS:
        if not np.array_equal(old[key][wanted],expected[key]):raise RuntimeError('FOUNDATION_TRADE_METRIC_CHANGED:'+key)
    values,tw_valid=c.values(candidate)
    checks=[]
    if c.references:
        events=np.concatenate(c.references)
        end=(max(references)+1)*timing.MINUTE_NS+timing.STUDY_START_NS
        prefix=features.L1FeatureCollectorV2(wanted,tick)
        prefix.consume(events[events[:,0]<end]);prefix.finish(end)
        pv,pm=prefix.values(candidate)
        compare=np.isin(wanted,sorted(references)[1:])&tw_valid
        if not compare.any():raise RuntimeError('FOUNDATION_PREFIX_NO_ADMITTED_ROWS')
        for key in values:
            if not np.allclose(values[key][compare],pv[key][compare],rtol=1e-12,atol=1e-10,equal_nan=True):raise RuntimeError('FOUNDATION_REAL_PREFIX:'+key)
        checks.append({'symbol':old['symbol'],'M1_rows':int(compare.sum()),'F_LAST_events':len(events),'prefix_to_M5_close':'PASS','future_events_used':False})
    return {'values':values,'ofi_valid':candidate,'bbo_valid':tw_valid,'covered_ns':c.covered,
       'first_mid':c.first,'last_mid':c.last,'files':old['files'],'prefix_checks':checks}


def build_levels_preflight_v2(plan,stage,native,start,contracts):
    volume_valid=native['m5_volume_valid'].to_numpy()
    # Reduce already bound real-trade ladders, summing fragments before profiles.
    histograms=[{} for _ in start]
    for symbol,item in plan['ladders'].items():
        previous.verify_digest(Path(item['path']),item['sha256'],'FOUNDATION_LADDER')
        df=pq.read_table(item['path']).to_pandas();df['bar_start_ns']=timing.STUDY_START_NS+(df.minute_index.to_numpy().astype(np.int64)//5)*5*timing.MINUTE_NS
        df['volume']=df.buy+df.sell+df.unknown
        sums=df.groupby(['bar_start_ns','price_tick'],sort=True).volume.sum()
        for stamp,h in sums.groupby(level=0,sort=True):
            row=int(np.searchsorted(start,stamp))
            if row>=len(start) or start[row]!=stamp or not volume_valid[row]:continue
            if histograms[row]:raise RuntimeError('FOUNDATION_CONTRACT_PROFILE_COLLISION')
            histograms[row]={int(p):int(v) for (_,p),v in h.items() if v>0}
            if sum(histograms[row].values())!=int(native['m5_volume'].to_numpy()[row]):raise RuntimeError('FOUNDATION_PROFILE_VOLUME_PARITY')
    levels=features.dynamic_profiles_v2(start,histograms,volume_valid,contracts,native['m5_close_usd'].to_numpy(),plan['tick_usd'])
    cut=len(start)//2
    prefix=features.dynamic_profiles_v2(start[:cut],histograms[:cut],volume_valid[:cut],contracts[:cut],native['m5_close_usd'].to_numpy()[:cut],plan['tick_usd'])
    for n,v in prefix.items():
        if not np.array_equal(v,levels[n][:cut],equal_nan=True):raise RuntimeError('FOUNDATION_PROFILE_PREFIX:'+n)
    table=pa.table({'m5_bar_start_ns':start,**levels})
    path=stage/'LEVELS.parquet';pq.write_table(table,path,compression='zstd')
    if not features.exact_feature_readback_v2(table,pq.read_table(path)):raise RuntimeError('FOUNDATION_LEVEL_READBACK')
    previous.write_json(stage/'LEVEL_CHECKS.json',{'prefix_rows':cut,'volume_reconciliation':'EXACT_EVERY_ADMITTED_M5','levels_sha256':previous.sha256_file(path)})
    print('FOUNDATION_REAL_LEVELS_AND_PREFIX_COMPLETE',flush=True)
    return levels,cut


def run_foundation_v2(binding_path):
    identity='GC_L1_DYNAMIC_FOUNDATION_FINALIZE_20261017_017';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('FOUNDATION_BINDING')
    binding=previous.read_json(binding_path);path=REPO/'configs/research'/f'{identity}.json'
    previous.verify_digest(path,binding['plan_sha256'],'FOUNDATION_PLAN');plan=previous.read_json(path)
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'FOUNDATION_POLICY')
    previous.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'FOUNDATION_APPROVAL')
    approval=previous.read_json(root/'OPERATOR_APPROVAL.json');policy=previous.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_l1_dynamic_foundation_017']
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['max_invocations']!=1 or previous.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):raise RuntimeError('FOUNDATION_AUTHORITY')
    for rel,digest in binding['code_sha256'].items():previous.verify_digest(REPO/rel,digest,'FOUNDATION_CODE')
    for item in plan['inputs'].values():previous.verify_digest(Path(item['path']),item['sha256'],'FOUNDATION_INPUT')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False)
    previous.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head']})
    base=previous.read_json(Path(plan['inputs']['source_plan']['path']));groups,hashes=previous._bound_gc_inputs(base)
    old=pq.read_table(plan['inputs']['trade_masks']['path']);q=pq.read_table(plan['inputs']['L1_candidates']['path'])
    native=pq.read_table(plan['inputs']['M5_features']['path']);roll=previous.read_json(Path(plan['inputs']['roll']['path']))
    minute=old['bar_start_ns'].to_numpy();contract=old['contract_index'].fill_null(-1).to_numpy();raw=(minute-timing.STUDY_START_NS)//timing.MINUTE_NS
    if not np.array_equal(minute,q['bar_start_ns'].to_numpy()) or not np.array_equal(contract,q['contract_index'].fill_null(-1).to_numpy()):raise RuntimeError('FOUNDATION_CANDIDATE_CLOCK')
    candidate=q['ofi_event_candidate_valid'].to_numpy();bbo_candidate=q['bbo_time_integral_candidate_valid'].to_numpy()
    if not np.array_equal(candidate,bbo_candidate):raise RuntimeError('FOUNDATION_012_MASK_VERSION_CHANGED')
    start=native['m5_bar_start_ns'].to_numpy();query=start[:,None]+np.arange(5)[None,:]*timing.MINUTE_NS
    pos=np.searchsorted(minute,query);idx=np.minimum(pos,len(minute)-1)
    present=(pos<len(minute))&(minute[idx]==query);same=present.all(axis=1)&(contract[idx]>=0).all(axis=1)&(contract[idx]==contract[idx[:,:1]]).all(axis=1)
    m5candidate=same&candidate[idx].all(axis=1)
    if int(m5candidate.sum())!=plan['expected_M5_candidates']:raise RuntimeError('FOUNDATION_CANDIDATE_SUPPORT')
    levels,cut=build_levels_preflight_v2(plan,stage,native,start,contract[idx[:,0]])
    results=[];columns={n:np.full(len(minute),np.nan) for n in features.L1_FIELDS_V2};masks={n:np.zeros(len(minute),bool) for n in ('ofi','bbo')}
    coverage_ns=np.zeros(len(minute),np.int64);first_mid=np.full(len(minute),np.nan);last_mid=first_mid.copy();checks=[];records=0
    with ProcessPoolExecutor(max_workers=plan['worker_count'],mp_context=mp.get_context('fork')) as pool:
        for group in groups:
            ci=roll['contract_order'].index(group['raw_symbol']);rows=np.flatnonzero(contract==ci)
            ref=np.flatnonzero(m5candidate&(contract[idx[:,0]]==ci))
            refs=[] if not len(ref) else list(range(int((start[ref[0]]-timing.STUDY_START_NS)//timing.MINUTE_NS)-1,int((start[ref[0]]-timing.STUDY_START_NS)//timing.MINUTE_NS)+5))
            expected={key:old[key].to_numpy()[rows] for key in source.METRICS}
            results.append((rows,pool.submit(foundation_worker_v2,group,roll['metadata_identity'][group['raw_symbol']][2],raw[rows],candidate[rows],expected,refs)))
        for rows,future in results:
            r=future.result()
            for n,v in r['values'].items():columns[n][rows]=v
            for n in masks:masks[n][rows]=r[n+'_valid']
            coverage_ns[rows]=r['covered_ns'];first_mid[rows]=r['first_mid'];last_mid[rows]=r['last_mid']
            checks.extend(r['prefix_checks']);records+=sum(f['records'] for f in r['files'])
    if records!=base['expected_source_records']:raise RuntimeError('FOUNDATION_RAW_COUNT')
    mt=pa.table({'bar_start_ns':minute,'contract_index':contract,'ofi_valid':masks['ofi'],'bbo_valid':masks['bbo'],'integral_covered_ns':coverage_ns,'opening_mid_tick':first_mid,'closing_mid_tick':last_mid,**columns})
    pq.write_table(mt,stage/'M1_L1_FEATURES.parquet',compression='zstd')
    previous.write_json(stage/'L1_CHECKS.json',{'raw_records':records,'real_L1_prefix_checks':checks,'M1_sha256':previous.sha256_file(stage/'M1_L1_FEATURES.parquet')})
    if np.any(coverage_ns[candidate]!=timing.MINUTE_NS):
        # OFI and integral masks stay separate; partial integrals never qualify.
        print('FOUNDATION_PARTIAL_INTEGRALS',int(np.count_nonzero(candidate&(coverage_ns!=timing.MINUTE_NS))),flush=True)
    out={'m5_bar_start_ns':start,'decision_ns':native['decision_ns'].to_numpy(),'contract_index':contract[idx[:,0]]}
    for n,v in masks.items():out[n+'_valid']=same&v[idx].all(axis=1)
    for j,n in enumerate(features.L1_FIELDS_V2):
        v=columns[n][idx].sum(axis=1) if j<6 else columns[n][idx].mean(axis=1)
        if n=='mid_response_ticks':v=last_mid[idx[:,-1]]-first_mid[idx[:,0]]
        valid=out['ofi_valid'] if j<6 else out['bbo_valid'];v[~valid]=np.nan;out[n]=v
    # Existing trusted-trade intensity has its own unchanged unsigned mask.
    volume_valid=native['m5_volume_valid'].to_numpy();out['trade_intensity_valid']=volume_valid
    out['trade_records_per_second']=np.where(volume_valid,old['trades'].to_numpy()[idx].sum(axis=1)/300,np.nan)
    out.update(levels)
    previous.write_json(stage/'PREPUBLICATION_CHECKS.json',{'raw_records':records,'real_L1_prefix_checks':checks,'profile_prefix_rows':cut,'volume_reconciliation':'EXACT_EVERY_ADMITTED_M5'})
    table=pa.table(out);output=stage/'M5_L1_DYNAMIC_FEATURES.parquet';pq.write_table(table,output,compression='zstd')
    if not features.exact_feature_readback_v2(table,pq.read_table(output)):raise RuntimeError('FOUNDATION_READBACK')
    months=pd.DatetimeIndex(start,tz='UTC').strftime('%Y-%m').to_numpy()
    masknames=[n for n in out if n.endswith('_valid')]
    counts={n:{'rows':int(out[n].sum()),'by_month':{m:int(np.count_nonzero(out[n]&(months==m))) for m in sorted(set(months))}} for n in masknames}
    previous._rehash_sources(hashes,[])
    for item in list(plan['inputs'].values())+list(plan['ladders'].values()):previous.verify_digest(Path(item['path']),item['sha256'],'FOUNDATION_INPUT_POST')
    for rel,digest in binding['code_sha256'].items():previous.verify_digest(REPO/rel,digest,'FOUNDATION_CODE_POST')
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'FOUNDATION_POLICY_POST')
    if previous.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('FOUNDATION_SOURCE_CHANGED')
    result={'status':'MASKED_L1_AND_DYNAMIC_PROFILES_BUILT','source_head':binding['source_head'],'limits':proof,
       'raw_records':records,'native_M5_rows':len(start),'candidate_M5_rows':int(m5candidate.sum()),'coverage':counts,
       'real_L1_prefix_checks':checks,'real_profile_prefix_rows':cut,'ladder_volume_reconciliation':'EXACT_EVERY_ADMITTED_M5',
       'session_observed_is_incomplete_diagnostic':True,'session_holiday_calendar_qualified':False,
       'M5_file_sha256':previous.sha256_file(output),'TEST_accessed':False,'fit_started':False,'baseline_changed':False}
    previous.write_json(stage/'RESULT.json',result)
    previous.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':previous.sha256_file(stage/'RESULT.json')})
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('GC_L1_DYNAMIC_FOUNDATION_COMPLETE',flush=True)


def full_profile_worker_v2(group,ci,selected,tick,stage):
    path=stage/(group['raw_symbol']+'.parquet')
    c=features.TradeLadderCollector(path,selected,np.ones(len(selected),bool),ci,tick)
    try:r=source.scan_symbol(group['items'],tick,on_batch=c)
    finally:c.close()
    rows=np.flatnonzero(selected==ci)
    return {'rows':rows,'metrics':{k:r[k][rows] for k in source.METRICS},'ohlc':c.ohlc[rows],
       'files':r['files'],'path':str(path),'sha256':previous.sha256_file(path)}


def run_full_history_profiles_v2(binding_path):
    identity='GC_FULL_HISTORY_PROFILES_REPAIR_20261017_019';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('FULL_PROFILE_BINDING')
    binding=previous.read_json(binding_path);rel='configs/research/'+identity+'.json';previous.verify_digest(REPO/rel,binding['plan_sha256'],'FULL_PROFILE_PLAN');plan=previous.read_json(REPO/rel)
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'FULL_PROFILE_POLICY');previous.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'FULL_PROFILE_APPROVAL')
    policy=previous.read_json(REPO/'NEXT_RUN_POLICY.json');approval=previous.read_json(root/'OPERATOR_APPROVAL.json');gate=policy['current_work']['gc_full_history_profiles_019']
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['max_invocations']!=1 or approval['approved_plan_sha256']!=binding['plan_sha256'] or previous.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):raise RuntimeError('FULL_PROFILE_AUTHORITY')
    for p,h in binding['code_sha256'].items():previous.verify_digest(REPO/p,h,'FULL_PROFILE_CODE')
    for b in plan['inputs'].values():previous.verify_digest(Path(b['path']),b['sha256'],'FULL_PROFILE_INPUT')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry();stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False);ladder=stage/'LADDERS';ladder.mkdir()
    previous.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head']})
    base=previous.read_json(Path(plan['inputs']['source_plan']['path']));groups,hashes=previous._bound_gc_inputs(base)
    roll=previous.read_json(Path(plan['inputs']['roll']['path']));old=pq.read_table(plan['inputs']['trade_masks']['path'])
    width=(timing.WINDOW_END_NS-timing.STUDY_START_NS)//timing.MINUTE_NS;minutes=timing.STUDY_START_NS+np.arange(width,dtype=np.int64)*timing.MINUTE_NS
    selected=features.full_gc_minute_selection_v2(minutes,roll)
    oldidx=(old['bar_start_ns'].to_numpy()-timing.STUDY_START_NS)//timing.MINUTE_NS
    if not np.array_equal(selected[oldidx],old['contract_index'].fill_null(-1).to_numpy()):raise RuntimeError('FULL_PROFILE_CHANGED_004_SELECTION')
    metrics={k:np.zeros(width,np.uint64) for k in source.METRICS};ohlc=np.zeros((width,4),np.int64);paths={};records=0
    with ProcessPoolExecutor(max_workers=plan['worker_count'],mp_context=mp.get_context('fork')) as pool:
        jobs=[pool.submit(full_profile_worker_v2,g,roll['contract_order'].index(g['raw_symbol']),selected,roll['metadata_identity'][g['raw_symbol']][2],ladder) for g in groups]
        for job in jobs:
            r=job.result();rows=r['rows']
            for k in metrics:metrics[k][rows]=r['metrics'][k]
            ohlc[rows]=r['ohlc'];paths[r['path']]=r['sha256'];records+=sum(f['records'] for f in r['files'])
    if records!=base['expected_source_records']:raise RuntimeError('FULL_PROFILE_RAW_COUNT')
    for k in metrics:
        if not np.array_equal(metrics[k][oldidx],old[k].to_numpy()):raise RuntimeError('FULL_PROFILE_CHANGED_004_TRADE_METRIC:'+k)
    eligible=minutes+timing.MINUTE_NS<timing.WINDOW_END_NS
    masks=source.row_masks(selected,metrics,eligible);valid=masks['volume_profile'].reshape(-1,5).all(axis=1)
    contracts=selected.reshape(-1,5);valid &=(contracts>=0).all(axis=1)&(contracts==contracts[:,:1]).all(axis=1)
    contracts=contracts[:,0];start=minutes[::5];volume=metrics['volume'].reshape(-1,5).sum(axis=1)
    cache=pa.table({'bar_start_ns':minutes,'contract_index':selected,'volume_profile_valid':masks['volume_profile'],**metrics,**{n:ohlc[:,j] for j,n in enumerate(['open_tick','high_tick','low_tick','close_tick'])}})
    pq.write_table(cache,stage/'FULL_GC_M1_TRADE_INPUTS.parquet',compression='zstd')
    previous.write_json(stage/'SOURCE_CHECKS.json',{'records':records,'old_004_metric_parity':'EXACT_EVERY_NATIVE_ELIGIBLE_M1','ladder_sha256':paths})
    histograms=[{} for _ in start]
    for path,digest in paths.items():
        previous.verify_digest(Path(path),digest,'FULL_PROFILE_LADDER')
        df=pq.read_table(path).to_pandas();df['bar']=df.minute_index.to_numpy().astype(np.int64)//5;df['volume']=df.buy+df.sell+df.unknown
        sums=df.groupby(['bar','price_tick'],sort=True).volume.sum()
        for row,h in sums.groupby(level=0,sort=True):
            row=int(row)
            if not valid[row]:continue
            if histograms[row]:raise RuntimeError('FULL_PROFILE_CONTRACT_COLLISION')
            histograms[row]={int(p):int(v) for (_,p),v in h.items() if v>0}
            if sum(histograms[row].values())!=int(volume[row]):raise RuntimeError('FULL_PROFILE_VOLUME_PARITY')
    close=ohlc[4::5,3].astype(float)*plan['tick_usd']
    values=features.dynamic_profiles_v2(start,histograms,valid,contracts,close,plan['tick_usd'])
    cut=len(start)//2;prefix=features.dynamic_profiles_v2(start[:cut],histograms[:cut],valid[:cut],contracts[:cut],close[:cut],plan['tick_usd'])
    for key in values:
        if not np.array_equal(values[key][:cut],prefix[key],equal_nan=True):raise RuntimeError('FULL_PROFILE_PREFIX:'+key)
    native=pq.read_table(plan['inputs']['M5_features']['path']);nstart=native['m5_bar_start_ns'].to_numpy();index=np.searchsorted(start,nstart)
    if np.any(index>=len(start)) or not np.array_equal(start[index],nstart) or np.any(nstart+5*timing.MINUTE_NS>=timing.WINDOW_END_NS):raise RuntimeError('FULL_PROFILE_NATIVE_CLOCK')
    oldvalid=native['m5_volume_valid'].to_numpy()
    if not np.array_equal(volume[index][oldvalid],native['m5_volume'].to_numpy()[oldvalid]) or not np.array_equal(close[index][oldvalid],native['m5_close_usd'].to_numpy()[oldvalid]):raise RuntimeError('FULL_PROFILE_006_CONSERVATION')
    oldlevels=pq.read_table(plan['inputs']['prior_levels']['path']);overlap={}
    for family in ['rolling15','rolling60']:
        mask=oldlevels[family+'_valid'].to_numpy();newmask=values[family+'_valid'][index]
        if np.any(mask&~newmask):raise RuntimeError('FULL_PROFILE_LOST_PRIOR_QUALIFIED_ROWS')
        for level in ['vwap','poc','val','vah']:
            n=family+'_'+level+'_usd'
            if not np.array_equal(oldlevels[n].to_numpy()[mask],values[n][index][mask]):raise RuntimeError('FULL_PROFILE_PRIOR_VALUE_CHANGED:'+n)
        overlap[family]={'prior_valid':int(mask.sum()),'new_valid':int(newmask.sum()),'all_prior_values_exact':True}
    out={'m5_bar_start_ns':nstart,'decision_ns':native['decision_ns'].to_numpy(),'contract_index':contracts[index],'GC_volume_input_valid':valid[index],**{k:v[index] for k,v in values.items()}}
    table=pa.table(out);path=stage/'M5_FULL_GC_HISTORY_LEVELS.parquet';pq.write_table(table,path,compression='zstd')
    if not features.exact_feature_readback_v2(table,pq.read_table(path)):raise RuntimeError('FULL_PROFILE_READBACK')
    months=pd.DatetimeIndex(nstart,tz='UTC').strftime('%Y-%m').to_numpy();coverage={k:{'rows':int(v.sum()),'by_month':{m:int(np.count_nonzero(v&(months==m))) for m in sorted(set(months))}} for k,v in out.items() if k.endswith('_valid')}
    for b in plan['inputs'].values():previous.verify_digest(Path(b['path']),b['sha256'],'FULL_PROFILE_INPUT_POST')
    previous._rehash_sources(hashes,[])
    for p,h in binding['code_sha256'].items():previous.verify_digest(REPO/p,h,'FULL_PROFILE_CODE_POST')
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'FULL_PROFILE_POLICY_POST')
    if previous.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('FULL_PROFILE_SOURCE_CHANGED')
    result={'status':'FULL_GC_CLOCK_PROFILES_BUILT_UNCHANGED_TRADE_QUALITY','source_head':binding['source_head'],'limits':proof,'raw_records':records,'native_rows':len(nstart),'full_GC_M1_rows':width,'coverage':coverage,'prior_rolling_overlap':overlap,'prefix_full_GC_M5_rows':cut,'native_clock_gating_removed_from_history_only':True,'original_masks_or_baseline_changed':False,'TEST_accessed':False,'fit_started':False,'session_holiday_calendar_qualified':False,'profile_file_sha256':previous.sha256_file(path)}
    previous.write_json(stage/'RESULT.json',result);previous.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':previous.sha256_file(stage/'RESULT.json')})
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('GC_FULL_HISTORY_PROFILES_COMPLETE',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--binding',type=Path)
    group.add_argument('--foundation-binding',type=Path)
    group.add_argument('--full-history-binding',type=Path)
    args = parser.parse_args()
    if args.full_history_binding:
        run_full_history_profiles_v2(args.full_history_binding)
        return
    if args.foundation_binding:
        run_foundation_v2(args.foundation_binding)
        return
    try:
        run(args.binding)
    except Exception as exc:
        if ROOT.is_dir() and not (ROOT/'FAILURE.json').exists():
            previous.write_json(ROOT/'FAILURE.json',{'status':'FAILED_NO_ADMISSION','error':str(exc),
                'exception':type(exc).__name__,'failed_utc':datetime.now(timezone.utc).isoformat(),
                'test_accessed':False,'fit_started':False})
        raise


if __name__=='__main__':
    main()
