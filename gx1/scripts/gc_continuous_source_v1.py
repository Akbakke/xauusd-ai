"""Execute one frozen, local-only continuous GC source-admission plan."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gx1.contracts import gc_continuous_source_v1 as contract
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.scripts import gc_source_disposition_v1 as previous

REPO = previous.REPO
PLAN_ID = 'GC_CONTINUOUS_SOURCE_ADMISSION_20261016_004'
PLAN_REL = f'configs/research/{PLAN_ID}.json'
ROOT = Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001') / PLAN_ID
CODE_PATHS = (
    'gx1/contracts/gc_continuous_source_v1.py', 'gx1/scripts/gc_continuous_source_v1.py',
    'tests/test_gc_continuous_source_v1.py',
    'gx1/contracts/gc_source_disposition_v1.py', 'gx1/scripts/gc_source_disposition_v1.py',
    'gx1/contracts/gc_order_flow_source_v1.py', 'gx1/scripts/research_ta_campaign_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
)
WORKER_COUNT = 8


def runtime(binding_path: Path):
    if binding_path != ROOT/'EXECUTION_BINDING.json':
        raise RuntimeError('GC_CONTINUOUS_BINDING_PATH_INVALID')
    previous.require_regular_file(binding_path, 'BINDING')
    binding = previous.read_json(binding_path)
    previous.verify_digest(REPO/PLAN_REL, binding['plan_sha256'], 'PLAN')
    plan = previous.read_json(REPO/PLAN_REL)
    previous.verify_digest(ROOT/'OPERATOR_APPROVAL.json', binding['approval_sha256'], 'APPROVAL')
    approval = previous.read_json(ROOT/'OPERATOR_APPROVAL.json')
    previous.verify_digest(REPO/'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'POLICY')
    policy = previous.read_json(REPO/'NEXT_RUN_POLICY.json')
    gate = policy['current_work']['gc_continuous_source_plan_004']
    if (plan['plan_id'] != PLAN_ID or binding['source_head'] != previous.git_head()
            or approval['approved_plan_sha256'] != binding['plan_sha256']
            or approval['single_use'] is not True or approval['max_invocations'] != 1
            or approval['status'] != 'APPROVED_SINGLE_USE_PENDING_EXECUTION'
            or gate['status'] != 'EXECUTION_CLAIMED' or gate['plan_sha256'] != binding['plan_sha256']
            or gate['completed_invocations'] != 0 or gate['claimed_invocations'] != 1
            or gate['execution_authorized'] is not False
            or policy['training_enabled'] is not False):
        raise RuntimeError('GC_CONTINUOUS_SINGLE_USE_AUTHORITY_INVALID')
    if set(binding['code_sha256']) != set(CODE_PATHS):
        raise RuntimeError('GC_CONTINUOUS_CODE_INVENTORY_INVALID')
    for path,digest in binding['code_sha256'].items():
        previous.verify_digest(REPO/path,digest,'CODE')
    if subprocess.run(['git','status','--porcelain'],cwd=REPO,check=True,capture_output=True,text=True).stdout:
        raise RuntimeError('GC_CONTINUOUS_DIRTY_SOURCE')
    if subprocess.run(['git','branch','--show-current'],cwd=REPO,check=True,capture_output=True,text=True).stdout.strip() != 'work/gx1-current':
        raise RuntimeError('GC_CONTINUOUS_BRANCH_INVALID')
    proof = require_capped_cpu_audit_execution()
    proof['lock'] = require_capped_lock_ancestry()
    if proof['cpu_affinity'] != list(range(8)):
        raise RuntimeError('GC_CONTINUOUS_AFFINITY_INVALID')
    if (ROOT/'RESULTS').exists() or (ROOT/'.RESULTS.staging').exists():
        raise RuntimeError('GC_CONTINUOUS_OUTPUT_ALREADY_EXISTS')
    previous.write_json(ROOT/'EXECUTION_CLAIM.json', {
        'status':'CLAIMED','invocation':1,'plan_sha256':binding['plan_sha256'],
        'binding_sha256':previous.sha256_file(binding_path),
        'source_head':binding['source_head'],'claimed_utc':datetime.now(timezone.utc).isoformat()})
    return plan,binding,proof


def load_metadata(plan, groups):
    item = plan['inputs']['source_inventory']
    path = Path(item['path'])
    previous.verify_digest(path,item['sha256'],'SOURCE_INVENTORY')
    inventory = previous.read_json(path)
    identities = {g['raw_symbol']:(g['items'][0]['instrument_id'],g['items'][0]['publisher_id']) for g in groups}
    metadata = {s:set() for s in identities}
    hashes = [(path,item['sha256'])]
    for source in inventory['source_files']:
        if source['schema'] != 'definition':
            continue
        path,receipt = Path(source['local_file_path']),Path(source['receipt_path'])
        previous.verify_digest(path,source['local_file_sha256_expected_from_receipt'],'DEFINITION')
        previous.verify_digest(receipt,source['receipt_sha256'],'DEFINITION_RECEIPT')
        if path.stat().st_size != source['local_file_bytes'] or path.is_relative_to(previous.GC_ROOT) is not True:
            raise RuntimeError('GC_CONTINUOUS_DEFINITION_PATH_OR_SIZE_INVALID')
        hashes.extend(((path,source['local_file_sha256_expected_from_receipt']),(receipt,source['receipt_sha256'])))
        with gzip.open(path,'rt',newline='') as stream:
            for row in csv.DictReader(stream):
                symbol = row['raw_symbol']
                if symbol != source['raw_symbol'] or (int(row['instrument_id']),int(row['publisher_id'])) != identities[symbol]:
                    raise RuntimeError('GC_CONTINUOUS_DEFINITION_IDENTITY_INVALID')
                metadata[symbol].add((int(row['maturity_year']),int(row['maturity_month']),int(row['min_price_increment'])))
    if any(len(v) != 1 for v in metadata.values()):
        raise RuntimeError('GC_CONTINUOUS_NONCONSTANT_CONTRACT_METADATA')
    result = {s:next(iter(v)) for s,v in metadata.items()}
    if any(tick <= 0 or not 1 <= month <= 12 for year,month,tick in result.values()):
        raise RuntimeError('GC_CONTINUOUS_INVALID_CONTRACT_METADATA')
    return result,hashes


def build_series(results, metadata, oanda_index):
    width = len(results[0]['volume'])
    minutes = timing.STUDY_START_NS + np.arange(width,dtype=np.int64)*timing.MINUTE_NS
    days = width//contract.MINUTES_PER_DAY
    daily = np.stack([r['volume'].reshape(days,contract.MINUTES_PER_DAY).sum(axis=1) for r in results])
    selected_daily,basis_daily = contract.choose_daily_contract(daily,[metadata[r['symbol']][:2] for r in results])
    prefix,_ = contract.choose_daily_contract(daily[:,:-1],[metadata[r['symbol']][:2] for r in results])
    if not np.array_equal(prefix,selected_daily[:-1]):
        raise RuntimeError('GC_CONTINUOUS_REAL_ROLL_PREFIX_MISMATCH')
    # Roll at 00:01 UTC, after the previous UTC day closes plus the existing 60s clock allowance.
    roll_day = (minutes - timing.STUDY_START_NS - timing.MINUTE_NS)//contract.DAY_NS
    selected = np.full(width,-1,dtype=np.int16)
    basis = np.full(width,-1,dtype=np.int32)
    ready = roll_day >= 0
    selected[ready] = selected_daily[roll_day[ready]]
    basis[ready] = basis_daily[roll_day[ready]]
    values = {name:np.zeros(width,dtype=np.uint64) for name in contract.METRICS}
    eligible = np.zeros(width,dtype=bool)
    for stamp in oanda_index:
        i=(stamp-timing.STUDY_START_NS)//timing.MINUTE_NS
        if 0 <= i < width and stamp+timing.MINUTE_NS < timing.WINDOW_END_NS:
            eligible[i]=True
    for i,result in enumerate(results):
        take=selected==i
        for name in contract.METRICS:
            values[name][take]=result[name][take]
    if not np.array_equal(values['volume'],values['buy']+values['sell']+values['unknown']):
        raise RuntimeError('GC_CONTINUOUS_VOLUME_CONSERVATION_FAILED')
    chosen=selected>=0
    basis_available=timing.STUDY_START_NS+(basis.astype(np.int64)+1)*contract.DAY_NS+timing.MINUTE_NS
    if np.any(basis_available[chosen]>minutes[chosen]):
        raise RuntimeError('GC_CONTINUOUS_ROLL_BASIS_NOT_AVAILABLE')
    masks=contract.row_masks(selected,values,eligible)
    segments=contract.reset_segments(selected,masks['cvd'],minutes)
    bad_clock_trades=0
    for i,result in enumerate(results):
        for f in result['files']:
            start,end=f['start_ns'],f['end_ns']
            lo=(start-timing.STUDY_START_NS)//timing.MINUTE_NS
            hi=(end-timing.STUDY_START_NS)//timing.MINUTE_NS
            if np.any(selected[lo:hi]==i):
                bad_clock_trades+=f['bad_recv_trades']
    months=minutes.astype('datetime64[ns]').astype('datetime64[M]')
    decisions=contract.admission_decisions(months,masks,bad_clock_trades)
    # Measure whole closed M5 windows using the same M1 availability clock.
    m5={}
    for name,mask in masks.items():
        blocks=mask.reshape(-1,5).all(axis=1)
        chosen=selected.reshape(-1,5)
        blocks &= (chosen==chosen[:,0,None]).all(axis=1)
        m5[name]=int(blocks.sum())
    table=pa.table({
        'bar_start_ns':minutes[eligible], 'available_ns':(minutes+timing.MINUTE_NS)[eligible],
        'contract_index':pa.array(selected[eligible],mask=selected[eligible]<0),
        'roll_basis_utc_day_index':pa.array(basis[eligible],mask=basis[eligible]<0),
        **{name:array[eligible] for name,array in values.items()},
        **{f'{name}_input_valid':array[eligible] for name,array in masks.items()},
        'cvd_segment_id':pa.array(segments[eligible],mask=segments[eligible]<0),
    })
    schedule=[]
    for d in range(days):
        s=int(selected_daily[d])
        schedule.append({'roll_at_ns':int(timing.STUDY_START_NS+d*contract.DAY_NS+timing.MINUTE_NS),
            'selected_symbol':results[s]['symbol'] if s>=0 else None,
            'basis_day_index':int(basis_daily[d]),
            'previous_day_observed_volume_by_symbol':{r['symbol']:int(daily[i,d-1]) for i,r in enumerate(results)} if d else {}})
    summary={'oanda_m1_rows':int(eligible.sum()),'rows_with_contract':int(((selected>=0)&eligible).sum()),
        'rows_with_selected_gc_record':int(((values['records']>0)&eligible).sum()),
        'rows_with_selected_gc_trade':int(((values['trades']>0)&eligible).sum()),
        'potentially_selected_bad_receive_clock_trades':int(bad_clock_trades),
        'valid_closed_m5_windows':m5,'indicator_admission':decisions,
        'selected_observed_volume_contracts':int(values['volume'][eligible].sum()),
        'selected_unknown_volume_contracts':int(values['unknown'][eligible].sum()),
        'selected_quality_counts':{name:int(values[name][eligible].sum()) for name in
            ('gap','bad_recv_trade','invalid_trade','snapshot_trade','duplicate_trade')},
        'real_roll_prefix_check_passed':True,'integer_volume_conservation_passed':True,
        'source_complete':False,'historical_holiday_schedule':'SCHEDULE_UNKNOWN',
        'native_model_row_parity_proven':False,'feature_values_built':False,
        'test_accessed':False,'fit_started':False,'economics_evaluated':False}
    return table,schedule,summary


def run(binding_path):
    plan,binding,proof=runtime(binding_path)
    groups,gc_hashes=previous._bound_gc_inputs(plan)
    oanda_index,oanda_report,oanda_hashes=previous._load_oanda(plan)
    metadata,metadata_hashes=load_metadata(plan,groups)
    with ProcessPoolExecutor(max_workers=WORKER_COUNT,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(contract.scan_symbol,g['items'],metadata[g['raw_symbol']][2]) for g in groups]
        results=[f.result() for f in futures]
    if sum(f['records'] for r in results for f in r['files']) != plan['expected_source_records']:
        raise RuntimeError('GC_CONTINUOUS_STRUCTURAL_RECORD_RECONCILIATION_FAILED')
    old_result=previous.read_json(Path(plan['inputs']['gc_result']['path']))
    old_counts={f['path']:(f['counters']['records'],f['counters']['trade_records']) for f in old_result['files']}
    for r in results:
        for f in r['files']:
            if old_counts[f['path']] != (f['records'],f['raw_trade_records']):
                raise RuntimeError('GC_CONTINUOUS_PER_FILE_RECONCILIATION_FAILED')
    table,schedule,summary=build_series(results,metadata,oanda_index)
    previous._rehash_sources(gc_hashes,oanda_hashes)
    for path,digest in metadata_hashes:
        previous.verify_digest(path,digest,'METADATA_POST_SCAN')
    for path,digest in binding['code_sha256'].items():
        previous.verify_digest(REPO/path,digest,'CODE_POST_SCAN')
    for path,digest in ((REPO/PLAN_REL,binding['plan_sha256']),
            (REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256']),
            (ROOT/'OPERATOR_APPROVAL.json',binding['approval_sha256'])):
        previous.verify_digest(path,digest,'AUTHORITY_POST_SCAN')
    claim=previous.read_json(ROOT/'EXECUTION_CLAIM.json')
    previous.verify_digest(binding_path,claim['binding_sha256'],'BINDING_POST_SCAN')
    if previous.git_head()!=binding['source_head']:
        raise RuntimeError('GC_CONTINUOUS_HEAD_CHANGED')
    stage=ROOT/'.RESULTS.staging'
    stage.mkdir(exist_ok=False)
    dataset=stage/'SELECTED_M1_INPUT_MASKS.parquet'
    pq.write_table(table,dataset,compression='zstd')
    with dataset.open('rb') as f:
        os.fsync(f.fileno())
    reread=pq.read_table(dataset,use_threads=False)
    if not table.equals(reread):
        raise RuntimeError('GC_CONTINUOUS_PARQUET_PARITY_FAILED')
    previous.write_json(stage/'ROLL_SCHEDULE.json',{'contract_order':[r['symbol'] for r in results],
        'metadata_identity':metadata,'schedule':schedule})
    report={'status':'CONTINUOUS_SOURCE_ADMISSION_COMPLETE','plan_id':PLAN_ID,
        'plan_sha256':binding['plan_sha256'],'source_head':binding['source_head'],
        'code_sha256':binding['code_sha256'],'summary':summary,'oanda':oanda_report,
        'source_records':sum(f['records'] for r in results for f in r['files']),
        'source_files_rehashed':66,'definition_files_rehashed':(len(metadata_hashes)-1)//2,
        'source_file_measurements':[f for r in results for f in r['files']],
        'limits':proof,'new_databento_requests':0,'new_spending':False,
        'interpretation':plan['interpretation']}
    previous.write_json(stage/'RESULT.json',report)
    for path in stage.glob('*.json'):
        previous.read_json(path)
    inventory={p.name:{'sha256':previous.sha256_file(p),'bytes':p.stat().st_size} for p in stage.iterdir()}
    previous.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'invocation':1,
        'plan_sha256':binding['plan_sha256'],'artifacts':inventory,
        'test_accessed':False,'fit_started':False,'model_admitted':False})
    if set(p.name for p in stage.iterdir()) != set(inventory)|{'TERMINAL.json'}:
        raise RuntimeError('GC_CONTINUOUS_PUBLICATION_INVENTORY_MISMATCH')
    fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    publish_bundle_directory_noreplace(stage,ROOT/'RESULTS')
    print(json.dumps(summary,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.binding)
    except Exception as exc:
        failure = ROOT/'FAILURE.json'
        if ROOT.is_dir() and not failure.exists():
            previous.write_json(failure,{'status':'FAILED_NO_ADMISSION','error':str(exc),
                'exception':type(exc).__name__,'failed_utc':datetime.now(timezone.utc).isoformat(),
                'test_accessed':False,'fit_started':False})
        raise

if __name__=='__main__':
    main()
