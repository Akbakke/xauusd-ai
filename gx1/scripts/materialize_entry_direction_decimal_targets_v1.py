"""Create a separately bound exact-decimal 1..60 target, never fit or trade."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO=Path('/home/andre2/src/GX1_CURRENT')
ROOT=Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE='independent_direction_decimal_targets_actual_20261028_001'
SOURCE_PATHS=(
    'gx1/scripts/materialize_entry_direction_decimal_targets_v1.py',
    'gx1/scripts/materialize_entry_direction_dense_v1.py',
    'gx1/scripts/audit_entry_direction_decimal_sign_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/xau_tape_provenance_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_materialize_entry_direction_decimal_targets_v1.py',
)


def require(ok,reason):
    if not bool(ok):raise RuntimeError('DIRECTION_DECIMAL_TARGETS_'+reason)


def reconstruct_grid(source,numerators,precision):
    """Recover ticks only when original decimal midpoint and float sides both agree.

    Caller supplies the common-grid numerator independently derived from every
    bound vendor decimal string; float round-trip alone is not source proof.
    """
    from gx1.scripts.materialize_entry_direction_dense_v1 import DECIMAL_GRID_SCHEMA
    require(type(precision) is int and 0<=precision<=12 and isinstance(numerators,np.ndarray)
        and numerators.dtype==np.int64 and numerators.ndim==1 and len(numerators)>0,'GRID_HEADER')
    scale=10**precision;out=[]
    for name in ('bid_open','ask_open'):
        values=source[name]
        require(isinstance(values,np.ndarray) and values.dtype==np.float64 and values.shape==numerators.shape
            and np.isfinite(values).all() and np.all(values>0),'GRID_PRICE:'+name)
        rounded=np.rint(values*scale)
        require(np.isfinite(rounded).all() and np.all((rounded>0)&(rounded<2**50)),'GRID_BOUND:'+name)
        ticks=rounded.astype(np.int64)
        require(np.array_equal(ticks.astype(np.float64)/scale,values),'GRID_FLOAT_PARITY:'+name)
        ticks.setflags(write=False);out.append(ticks)
    require(np.all(out[0]<=out[1]) and np.array_equal(out[0]+out[1],numerators),'GRID_DECIMAL_SOURCE_PARITY')
    return {'schema_version':DECIMAL_GRID_SCHEMA,'decimal_places':precision,'bid_ticks':out[0],'ask_ticks':out[1]}


def require_corrected_target(old,new,audit_result,*,decimal):
    """Check all saved clocks/masks and the independent 60-horizon sign matrix."""
    for name in ('parent_rows','bar_start_ns','decision_ns','dynamic_entry_m1_row','terminal_time_ns','terminal_source_row','observed'):
        require(old[name].dtype==new[name].dtype and np.array_equal(old[name],new[name]),'PRESERVE:'+name)
    old_sign,new_sign,obs,terminal=old['sign'],new['sign'],old['observed'],old['terminal_time_ns']
    groups={'all_118249':decimal.confusion_by_horizon(old_sign,new_sign,obs,terminal<decimal.CUTOFF_NS),
        'fold1_train':decimal.confusion_by_horizon(old_sign[:decimal.FOLD_TRAIN],new_sign[:decimal.FOLD_TRAIN],
            obs[:decimal.FOLD_TRAIN],terminal[:decimal.FOLD_TRAIN]<decimal.TRAIN_CUTOFF),
        'fold1_later':decimal.confusion_by_horizon(old_sign[decimal.FOLD_TRAIN:decimal.FOLD_LATER_END],new_sign[decimal.FOLD_TRAIN:decimal.FOLD_LATER_END],
            obs[decimal.FOLD_TRAIN:decimal.FOLD_LATER_END],terminal[decimal.FOLD_TRAIN:decimal.FOLD_LATER_END]<decimal.LATER_CUTOFF)}
    require(audit_result['status']=='COMPLETE' and audit_result['test_data_used'] is False
        and groups==audit_result['confusions'],'AUDIT_CONFUSION_PARITY')
    require(np.array_equal(np.sign(new['mid_move']).astype(np.int8),new_sign)
        and np.array_equal(np.sign(new['mid_return_bps']).astype(np.int8),new_sign)
        and np.all(new_sign[~obs]==0),'CORRECTED_SIGN_DOMAIN')
    return {'all_false_moves_removed':sum(x['saved_nonzero_to_exact_zero'] for x in groups['all_118249']),
        'nonzero_sign_flips':sum(x['nonzero_sign_flips'] for x in groups['all_118249'])}


def main(argv=None):
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','plan-sha256','policy-sha256','source-head'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    import pandas as pd
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import pyarrow.dataset
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.scripts import audit_entry_direction_decimal_sign_v1 as decimal
    from gx1.scripts import materialize_entry_direction_dense_v1 as dense
    require(not any(name.split('.')[0] in ('torch','sklearn','joblib') for name in sys.modules),'PRELOADED_MODEL')
    initial=candle.git_state();require(initial==(args.source_head,'work/gx1-current',''),'INITIAL_SOURCE')
    plan_ref={'path':str(candle.regular(args.plan)),'sha256':args.plan_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require(set(plan)=={'schema_version','stage','design','inputs','chunks','sources','output_dir','wall_seconds','test_data_used'}
            and plan['schema_version']=='gx1_direction_decimal_targets_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==600 and plan['test_data_used'] is False and len(plan['chunks'])==204,'PLAN')
        require(plan['design']['path']==str(REPO/'docs/DIRECTION_DECIMAL_TARGET_MATERIALIZATION_20261028.json'),'DESIGN_PATH')
        design=binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256'])
        require(design['schema_version']=='gx1_direction_decimal_targets_design_v1' and design['inputs']==plan['inputs']
            and design['test_data_used'] is False and design['parents']==decimal.PARENTS,'DESIGN')
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);gate=policy[STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False,'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled',
            'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')),'GLOBAL_GATES')
        require(set(plan['sources'])==set(SOURCE_PATHS)
            and all(ref['path']==str(REPO/name) for name,ref in plan['sources'].items()),'SOURCE_SET')
        for ref in plan['sources'].values():candle.regular(ref['path']);binding.require_binding(ref,label='source')
        require(set(plan['inputs'])=={'native_manifest','native_2024','native_2025','legacy_targets','dense_targets','audit_result'},'INPUT_SET')
        audit_ref=plan['inputs']['audit_result'];audit_result=binding.read_bound_json(Path(audit_ref['path']),audit_ref['sha256'])
        require(audit_result['status']=='COMPLETE' and audit_result['test_data_used'] is False
            and audit_result['source_chunks']==204 and audit_result['decimal_places_in_common_grid']==3,'AUDIT_REFERENCE')
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK_SCOPE')
    out=Path(plan['output_dir']);require(ROOT.resolve(strict=True)==ROOT and out.parent==ROOT
        and out.name=='DIRECTION_DECIMAL_TARGETS_20261028_001' and not out.exists() and not out.is_symlink(),'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads=[v for k,v in plan['inputs'].items() if k!='audit_result']+plan['chunks']
    require(len({x['path'] for x in payloads})==len(payloads),'DISTINCT_PAYLOADS')
    stage=out/'.OUTPUT.staging';target_staged=stage/'DENSE_DECIMAL_TARGETS.npz'
    metadata=[plan_ref['path'],policy_ref['path'],plan['design']['path'],audit_ref['path']]
    sys.addaudithook(candle.make_guard(out,[x['path'] for x in payloads],metadata,
        source_paths=[x['path'] for x in plan['sources'].values()],output_payload_paths=[str(target_staged)]))
    started=time.monotonic();previous_handler=signal.getsignal(signal.SIGALRM);previous_profile=sys.getprofile()
    def expired(*unused):raise RuntimeError('DIRECTION_DECIMAL_TARGETS_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(candle.no_queries_or_models)
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:candle.regular(ref['path']);binding.require_binding(ref,label='input')
            manifest=json.loads(Path(plan['inputs']['native_manifest']['path']).read_text())
            require(manifest['source_kind']=='oanda_native_mba_candles' and manifest['timestamp_semantics']=='bar_start_utc'
                and manifest['source_granularity']=='M1'
                and manifest['market_closure_contract']=='oanda_complete_true_scheduled_weekly_closure_excluded_v2','SOURCE_MANIFEST')
            expected,bounds=[],[];root=Path(manifest['out_root'])
            for item in manifest['source_chunks']:
                begin=int(pd.Timestamp(item['request_from_utc']).value);end=int(pd.Timestamp(item['request_to_utc_exclusive']).value)
                if end>decimal.START_NS and begin<decimal.CUTOFF_NS:
                    require(end<=decimal.TEST_NS,'CHUNK_REACHES_SEALED_TEST')
                    expected.append({'path':str(root/item['relative_path']),'sha256':item['sha256']});bounds.append((begin,end))
            require(expected==plan['chunks'],'EXACT_VENDOR_CHUNKS')
            years=[candle.read_year_quotes(plan['inputs']['native_'+str(year)]['path'],expected_rows=rows,year=year,
                parquet=parquet,arrow=arrow,np=np) for year,rows,offset in candle.YEAR_SPEC]
            full={key:np.concatenate([part[key] for part in years]) for key in years[0]}
            full['source_row']=np.concatenate([np.arange(offset,offset+rows,dtype=np.int64) for year,rows,offset in candle.YEAR_SPEC])
            times,rows,numerators,precision=decimal._read_raw_quotes(plan['chunks'],full,pd=pd,
                closure_mask=scheduled_weekly_closure_mask,request_bounds=bounds)
            selected=(full['time_ns']>=decimal.START_NS)&(full['time_ns']<decimal.CUTOFF_NS)
            source={key:np.ascontiguousarray(values[selected]) for key,values in full.items()}
            require(np.array_equal(times,source['time_ns']) and np.array_equal(rows,source['source_row'])
                and len(times)==audit_result['source_rows'] and precision==audit_result['decimal_places_in_common_grid'],'SOURCE_PARITY')
            grid=reconstruct_grid(source,numerators,precision)
            legacy=candle.read_targets(plan['inputs']['legacy_targets']['path'],np=np)
            corrected=dense.build_decimal_grid_dense_targets(source,legacy,decimal_grid=grid,cutoff_ns=decimal.CUTOFF_NS,block_rows=1024)
            old=decimal._read_target(plan['inputs']['dense_targets']['path'])
            parity=require_corrected_target(old,corrected,audit_result,decimal=decimal)
            require(parity=={'all_false_moves_removed':976,'nonzero_sign_flips':0},'AUDIT_COUNTS')
            input_snapshot=binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged();control_snapshot.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock,'FINAL_SOURCE_LOCK')
        with target_staged.open('xb') as handle:
            np.savez(handle,**corrected);handle.flush();os.fsync(handle.fileno())
        with np.load(target_staged,allow_pickle=False) as archive:
            require(set(archive.files)==set(corrected) and all(archive[k].dtype==corrected[k].dtype
                and np.array_equal(archive[k],corrected[k]) for k in corrected),'TARGET_ROUNDTRIP')
        digest=hashlib.sha256()
        with target_staged.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        target_ref={'path':str(out/'OUTPUT/DENSE_DECIMAL_TARGETS.npz'),'sha256':digest.hexdigest()}
        result={'schema_version':'gx1_direction_decimal_targets_result_v1','status':'COMPLETE','plan':plan_ref,
            'source_head':args.source_head,'target':target_ref,'target_identity':'exact_original_decimal_midpoint_v1',
            'parent_rows':decimal.PARENTS,'horizons_minutes':list(dense.HORIZONS),'decimal_places':precision,
            'source_rows':len(times),'source_clock_bid_ask_and_decimal_grid_parity':True,'independent_audit_confusions_reproduced':True,
            **parity,'scope':'New target archive only; original target and all model outputs preserved.',
            'limits':'No corrected-data model fit, baseline, predictive improvement, trading or economic edge demonstrated.',
            'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False}
        publish(stage/'RESULT.json',result)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
            'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT')
        publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False})
        raise
    finally:
        sys.setprofile(previous_profile);signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)


if __name__=='__main__':raise SystemExit(main())
