"""Independently verify the active 20-bar close-position formula on five clocks."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pandas as pd

from gx1.scripts.audit_direction_d1_trendline_replay_v1 import (
    CUTOFF_NS, compare_fields, load_m5_prefix, load_native_parts_prefix, window_exposure,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_close_position_numeric_20261028_001'
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
SHIFT_SECONDS = {'M5':300, 'M15':900, 'H1':3600, 'H4':14400, 'D1':86400}
SOURCE_PATHS = (
    'gx1/scripts/audit_direction_close_position_numeric_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_direction_close_position_numeric_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_CLOSE_POSITION_NUMERIC_' + reason)


def close_position_oracle(high, low, close):
    """NumPy 20-bar independent oracle; zero range stays unavailable."""
    high, low, close = (np.asarray(x, dtype=np.float64) for x in (high, low, close))
    require(high.ndim == low.ndim == close.ndim == 1 and high.shape == low.shape == close.shape
        and len(high) >= 20 and np.isfinite(high).all() and np.isfinite(low).all()
        and np.isfinite(close).all() and np.all(high >= low), 'ORACLE_DOMAIN')
    high20 = np.lib.stride_tricks.sliding_window_view(high, 20).max(axis=1)
    low20 = np.lib.stride_tricks.sliding_window_view(low, 20).min(axis=1)
    span = high20 - low20
    out = np.full(len(close), np.nan, dtype=np.float64)
    np.divide(close[19:] - low20, span, out=out[19:], where=span > 0)
    return out


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
    from gx1.features.entry_specialist_feature_groups_v1 import compile_gc_mtf_geometry_entry_execution_profile
    from gx1.features.htf_features import _resample_ohlcv, multi_tf_resample, build_multi_tf_v4_closed_timestamp_indices
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path':str(candle.regular(args.plan)), 'sha256':args.plan_sha256}
    policy_ref = {'path':str(REPO/'NEXT_RUN_POLICY.json'), 'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version','stage','design','inputs','sources','output_dir','wall_seconds','cutoff_ns','test_data_used'}
            and plan['schema_version']=='gx1_direction_close_position_numeric_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==1200 and plan['cutoff_ns']==CUTOFF_NS and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path']==str(REPO/'docs/DIRECTION_CLOSE_POSITION_NUMERIC_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version']=='gx1_direction_close_position_numeric_design_v1'
            and design['inputs']==plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate=policy[STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized','optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources'])==set(SOURCE_PATHS) and all(ref['path']==str(REPO/name) for name,ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values(): candle.regular(ref['path']);binding.require_binding(ref,label='source')
        keys={'source_m5','cache_manifest','prefix_normalization','fit_result','corrected_target','sample_pair','fold_design',
            'enriched_manifest','native_manifest','native_parts','cache_arrays','h1_native_reference'}
        require(set(plan['inputs'])==keys and set(plan['inputs']['cache_arrays'])==set(TIMEFRAMES), 'INPUT_SET')
        meta_names=('cache_manifest','prefix_normalization','fit_result','fold_design','enriched_manifest','native_manifest','h1_native_reference')
        meta={k:binding.read_bound_json(Path(plan['inputs'][k]['path']),plan['inputs'][k]['sha256']) for k in meta_names}
        manifest,norm,fit=(meta[k] for k in ('cache_manifest','prefix_normalization','fit_result'))
        previous=policy['independent_direction_h1_native_history_replay_20261028_001']
        require(previous['status']=='COMPLETE_ONE_SHOT_CONSUMED' and previous['result']==plan['inputs']['h1_native_reference']
            and meta['h1_native_reference']['replay_parity'] is True, 'PRIOR_NATIVE_PROOF')
        enriched=meta['enriched_manifest'];native_meta=enriched['native_m5_source'];parts=plan['inputs']['native_parts']
        expected_paths=[p for p in native_meta['part_paths'] if int(Path(p).parent.name.split('=')[1])<=2025]
        require(enriched['output_parquet']==plan['inputs']['source_m5']['path']
            and enriched['output_parquet_sha256']==plan['inputs']['source_m5']['sha256']
            and {'path':native_meta['manifest_path'],'sha256':native_meta['manifest_sha256']}==plan['inputs']['native_manifest']
            and isinstance(parts,list) and len(parts)==17 and [r['path'] for r in parts]==expected_paths
            and all(r['sha256']==native_meta['part_sha256'][r['path']] for r in parts), 'NATIVE_LINEAGE')
        require(fit['feature_view_identity']['prefix_normalization']['binding']==plan['inputs']['prefix_normalization'], 'FIT_BINDING')
        profile=compile_gc_mtf_geometry_entry_execution_profile(norm['gc_mtf_history']);field='close_position_range20'
        column=manifest['feature_names'].index(field)
        require(column in profile['mtf_indices_by_specialist']['chart_geometry_encoder'], 'ACTIVE_FIELD')
        cache_dir=Path(plan['inputs']['cache_manifest']['path']).parent
        for tf in TIMEFRAMES:
            item=manifest['tfs'][tf];refs=plan['inputs']['cache_arrays'][tf]
            require(set(refs)=={'features','timestamps'}
                and refs['features']=={'path':str(cache_dir/item['feats_npy']),'sha256':item['feats_npy_sha256']}
                and refs['timestamps']=={'path':str(cache_dir/item['ts_npy']),'sha256':item['ts_npy_sha256']}, 'CACHE_BINDING:'+tf)
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out=Path(plan['output_dir']);require(out.parent==ROOT and out.name=='DIRECTION_CLOSE_POSITION_NUMERIC_20261028_001'
        and ROOT.resolve(strict=True)==ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads=[plan['inputs'][k] for k in ('source_m5','corrected_target','sample_pair')]+parts
    for tf in TIMEFRAMES:payloads.extend(plan['inputs']['cache_arrays'][tf].values())
    metadata=[plan_ref['path'],policy_ref['path'],plan['design']['path']]+[plan['inputs'][k]['path'] for k in meta_names]
    sys.addaudithook(candle.make_guard(out,[ref['path'] for ref in payloads],metadata,source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM)
    def expired(*unused):raise RuntimeError('DIRECTION_CLOSE_POSITION_NUMERIC_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:candle.regular(ref['path']);binding.require_binding(ref,label='payload')
            enriched_frame,witness=load_m5_prefix(Path(plan['inputs']['source_m5']['path']),CUTOFF_NS)
            source=load_native_parts_prefix(parts,CUTOFF_NS);positions=source.index.get_indexer(enriched_frame.index)
            require(np.all(positions>=0) and np.array_equal(source.index.asi8[positions],enriched_frame.index.asi8)
                and np.array_equal(source.iloc[positions].to_numpy(),enriched_frame.to_numpy()), 'NATIVE_ENRICHED_OVERLAP')
            source_overlap=len(enriched_frame);del enriched_frame,positions
            with np.load(plan['inputs']['corrected_target']['path'],allow_pickle=False) as archive:
                parents=archive['parent_rows'].copy();decisions=archive['decision_ns'].copy()
            with np.load(plan['inputs']['sample_pair']['path'],allow_pickle=False) as archive:
                sample_parents=archive['parent_rows'].copy();sample_decisions=archive['decision_ns'].copy()
            fold=fit['fold'];later_fold=meta['fold_design']['folds'][1];positions=np.searchsorted(parents,sample_parents)
            require(parents.dtype==decisions.dtype==np.int64 and len(parents)==len(decisions)==118249
                and np.all(np.diff(parents)>0) and np.all(np.diff(decisions)>0)
                and np.searchsorted(decisions,fold['train_cutoff_ns'])==fold['oof_start_row']
                and np.searchsorted(decisions,fold['oof_cutoff_ns'])==fold['oof_end_row']
                and later_fold['oof_start_row']==fold['oof_end_row'] and later_fold['oof_cutoff_ns']==CUTOFF_NS
                and np.searchsorted(decisions,later_fold['oof_cutoff_ns'])==later_fold['oof_end_row']
                and len(sample_parents)==len(sample_decisions)==1760
                and np.all(positions>=later_fold['oof_start_row']) and np.all(positions<later_fold['oof_end_row'])
                and np.array_equal(parents[positions],sample_parents) and np.array_equal(decisions[positions],sample_decisions), 'DECISION_AXES')
            indices=build_multi_tf_v4_closed_timestamp_indices(source.index);outcomes={}
            agg={'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
            for tf in TIMEFRAMES:
                index=indices[tf]
                if tf=='M5':
                    require(index.equals(source.index), 'M5_AXIS');bars=source.loc[index,['open','high','low','close','volume']]
                elif tf in ('H1','M15'):
                    sample=source.iloc[:4096];expected=_resample_ohlcv(sample,tf).dropna(subset=['open','high','low','close'])
                    observed=multi_tf_resample(sample,tf).agg(agg).dropna(subset=['open','high','low','close'])
                    require(expected.index.equals(observed.index) and np.array_equal(expected.to_numpy(),observed.to_numpy()), 'FAST_RESAMPLE:'+tf)
                    bars=multi_tf_resample(source,tf).agg(agg).dropna(subset=['open','high','low','close']).loc[index]
                else:bars=_resample_ohlcv(source,tf).dropna(subset=['open','high','low','close']).loc[index]
                require(bars.index.equals(index), 'TF_AXIS:'+tf)
                oracle=close_position_oracle(bars['high'],bars['low'],bars['close'])
                refs=plan['inputs']['cache_arrays'][tf]
                with Path(refs['timestamps']['path']).open('rb') as handle:ts=np.load(handle,allow_pickle=False)
                cached=np.load(refs['features']['path'],mmap_mode='r',allow_pickle=False)
                trim=int(np.searchsorted(index.asi8,ts[0],side='left'));require(0<=trim<len(index) and index.asi8[trim]==ts[0], 'CACHE_START:'+tf)
                native_times=index.asi8[trim:];at=np.searchsorted(ts,native_times)
                require(np.array_equal(at,np.arange(len(at))) and np.array_equal(ts[at],native_times)
                    and cached.dtype==np.float32 and cached.shape==(manifest['tfs'][tf]['n_bars'],manifest['feature_count']), 'CACHE_AXIS:'+tf)
                actual=np.ascontiguousarray(oracle[trim:],dtype=np.float32);stored=np.ascontiguousarray(cached[at,column],dtype=np.float32)
                comparison=compare_fields(actual[:,None],stored[:,None],(field,),native_times)
                bad=(np.isnan(actual)!=np.isnan(stored)) | (~np.isnan(actual)&~np.isnan(stored)&(actual.view(np.uint32)!=stored.view(np.uint32)))
                shift=SHIFT_SECONDS[tf]*10**9;length=WINDOW_ROWS[tf]
                windows={'fit_train':window_exposure(native_times,bad,decisions[:fold['oof_start_row']],length,shift_ns=shift,timeframe=tf),
                    'fit_oof_q1':window_exposure(native_times,bad,decisions[fold['oof_start_row']:fold['oof_end_row']],length,shift_ns=shift,timeframe=tf),
                    'intervention_q2_sample':window_exposure(native_times,bad,sample_decisions,length,shift_ns=shift,timeframe=tf)}
                outcomes[tf]={'comparison':comparison,'window_exposure':windows,'source_rows':len(index),'window_rows':length}
                del bars,oracle,actual,stored,cached,bad
            payload_snapshot=binding.capture_verified_file_snapshot()
        payload_snapshot.require_unchanged();control_snapshot.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock
            and not any(n.split('.')[0] in ('torch','sklearn','joblib') for n in sys.modules),'FINAL_SOURCE_LOCK')
        passed=all(v['comparison']['mismatched_cells']==0 for v in outcomes.values())
        result={'schema_version':'gx1_direction_close_position_numeric_result_v1','status':'COMPLETE','plan':plan_ref,
            'source_head':args.source_head,'field':field,'independent_numpy_oracle':True,'native_enriched_exact_overlap_rows':source_overlap,
            'timeframes':outcomes,'all_five_exact':passed,
            'limits':'Independent formula arithmetic and observed input clock only; no proof of learned usefulness, generalization or economic edge.',
            'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
            'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT');publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')});return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':raise SystemExit(main())
