"""Research-only within-identical-M5-OFI rank audit; no data/fit authority.

Each pair compares later M1 OFI only within one exact physical float32 M5
OFI value. Block-pair summaries are pooled across those strata. This tests a
single temporal-order projection, not incremental value over all model inputs.
"""
from __future__ import annotations
import numpy as np
from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.scripts.score_entry_direction_dense_minutes_v1 import simultaneous_intervals


def _require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_M1_CONDITIONAL_' + reason)


def conditional_block_pairs(total, late, sign, eligible, block_ids):
    """Count exact within-M5-total positive/negative pairs by calendar blocks."""
    _require(isinstance(total,np.ndarray) and total.dtype==np.float32 and total.ndim==1
        and len(total)>0 and np.isfinite(total).all(), 'M5_TOTAL')
    n=len(total)
    _require(isinstance(late,np.ndarray) and late.dtype==np.float64 and late.shape==(n,)
        and np.isfinite(late).all() and isinstance(sign,np.ndarray) and sign.dtype==np.int8
        and sign.shape==(n,) and np.isin(sign,(-1,0,1)).all()
        and isinstance(eligible,np.ndarray) and eligible.dtype==np.bool_ and eligible.shape==(n,)
        and np.all(sign[eligible]!=0), 'INPUT_DOMAIN')
    _require(isinstance(block_ids,np.ndarray) and block_ids.dtype==np.int64 and block_ids.shape==(n,)
        and np.all(block_ids>=0) and np.array_equal(np.unique(block_ids),np.arange(int(block_ids.max())+1)), 'BLOCK_IDS')
    k=int(block_ids.max())+1
    wins=np.zeros((k,k),np.int64);pairs=np.zeros((k,k),np.int64)
    indices=np.flatnonzero(eligible);sorted_indices=indices[np.argsort(total[indices],kind='stable')]
    sorted_total=total[sorted_indices]
    split=np.r_[0,np.flatnonzero(sorted_total[1:]!=sorted_total[:-1])+1,len(sorted_indices)]
    matched_rows=matched_groups=0
    for left,right in zip(split[:-1],split[1:]):
        selected=sorted_indices[left:right];y=(sign[selected]==1).astype(np.int8)
        if not np.any(y==0) or not np.any(y==1):continue
        matched_rows+=len(selected);matched_groups+=1
        used,local=np.unique(block_ids[selected],return_inverse=True)
        summary=rank.block_rank_pairs(np.ascontiguousarray(y),np.ascontiguousarray(late[selected]),
            np.ascontiguousarray(local,dtype=np.int64))
        location=np.ix_(used,used)
        wins[location]+=summary['twice_wins']
        pairs[location]+=np.outer(summary['positive'],summary['negative'])
    count=int(pairs.sum());twice=int(wins.sum())
    _require(0<=twice<=2*count, 'PAIR_ACCOUNTING')
    return {'twice_wins':wins,'pairs':pairs,'eligible_rows':int(eligible.sum()),
        'matched_rows':int(matched_rows),'matched_groups':int(matched_groups),
        'pair_count':count,'auc':None if not count else twice/(2*count)}


def bootstrap_conditional_pairs(summary, weights):
    """Resample complete calendar blocks, keeping comparisons within M5 strata."""
    wins=summary['twice_wins'];pairs=summary['pairs'];w=np.asarray(weights,dtype=np.float64)
    _require(isinstance(wins,np.ndarray) and wins.dtype==np.int64 and wins.ndim==2
        and wins.shape==pairs.shape and wins.shape[0]==wins.shape[1]
        and isinstance(pairs,np.ndarray) and pairs.dtype==np.int64
        and w.ndim==2 and w.shape[1]==len(wins) and np.isfinite(w).all()
        and np.all(w>=0) and np.array_equal(w,np.floor(w)) and np.all(w.sum(axis=1)>0), 'BLOCK_WEIGHTS')
    numerator=np.sum((w@wins)*w,axis=1);denominator=np.sum((w@pairs)*w,axis=1)
    _require(np.all(numerator>=0) and np.all(numerator<=2*denominator), 'BOOTSTRAP_PAIR_ACCOUNTING')
    result=np.full(len(w),np.nan,np.float64);good=denominator>0
    result[good]=numerator[good]/(2*denominator[good])
    return result


def coarsen_conditional_pairs(summary,day_to_week):
    """Sum day-pair counts to week pairs without combining different M5 totals."""
    mapping=np.asarray(day_to_week)
    n=len(summary['pairs'])
    _require(mapping.dtype==np.int64 and mapping.shape==(n,) and np.all(mapping>=0), 'WEEK_MAP')
    weeks,inverse=np.unique(mapping,return_inverse=True)
    transform=np.zeros((n,len(weeks)),np.int64);transform[np.arange(n),inverse]=1
    return {'twice_wins':transform.T@summary['twice_wins']@transform,
        'pairs':transform.T@summary['pairs']@transform}


def conditional_simultaneous_bands(points,draws):
    """Use the existing 97.5% max-deviation owner; undefined stays undefined."""
    points=np.asarray(points,np.float64);draws=np.asarray(draws,np.float64)
    _require(points.ndim==1 and draws.ndim==2 and draws.shape[1]==len(points), 'BAND_SHAPE')
    if not (np.isfinite(points).all() and np.isfinite(draws).all()):return None
    return simultaneous_intervals(points,draws)


def conditional_multi_horizon(total, late, complete, decision_ns, terminal_ns, observed, sign,
    *, cutoff_ns, replicates, minimum_blocks, day_seed, week_seed):
    """Fixed-horizon, two-calendar-scheme simultaneous conditional rank audit."""
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights
    n=len(total);minute=60_000_000_000;day=86_400_000_000_000
    _require(isinstance(complete,np.ndarray) and complete.dtype==np.bool_ and complete.shape==(n,)
        and isinstance(decision_ns,np.ndarray) and decision_ns.dtype==np.int64 and decision_ns.shape==(n,)
        and np.all(np.diff(decision_ns)>0) and type(cutoff_ns) is int and np.all(decision_ns<cutoff_ns), 'PARENT_CLOCK')
    _require(isinstance(terminal_ns,np.ndarray) and terminal_ns.dtype==np.int64 and terminal_ns.ndim==2
        and terminal_ns.shape[0]==n and 0<terminal_ns.shape[1]<=60
        and isinstance(observed,np.ndarray) and observed.dtype==np.bool_ and observed.shape==terminal_ns.shape
        and isinstance(sign,np.ndarray) and sign.dtype==np.int8 and sign.shape==terminal_ns.shape
        and np.isin(sign,(-1,0,1)).all() and np.all(sign[~observed]==0), 'TARGET_DOMAIN')
    h=terminal_ns.shape[1]
    _require(np.array_equal(terminal_ns,decision_ns[:,None]+np.arange(1,h+1,dtype=np.int64)[None,:]*minute)
        and type(replicates) is int and replicates>0 and type(minimum_blocks) is int and minimum_blocks>0
        and type(day_seed) is int and type(week_seed) is int, 'TARGET_CLOCK_OR_POLICY')
    days=decision_ns//day;day_values,day_ids=np.unique(days,return_inverse=True)
    week_values=np.unique((day_values+3)//7);day_to_week=np.searchsorted(week_values,(day_values+3)//7).astype(np.int64)
    _require(len(day_values)>=minimum_blocks and len(week_values)>=minimum_blocks, 'MINIMUM_BLOCKS')
    dw=block_weights(len(day_values),seed=day_seed,np=np,replicates=replicates)
    ww=block_weights(len(week_values),seed=week_seed,np=np,replicates=replicates)
    points=np.full(h,np.nan,np.float64);day_draws=np.full((replicates,h),np.nan,np.float64)
    week_draws=np.full_like(day_draws,np.nan);panels=[]
    for i in range(h):
        mature=terminal_ns[:,i]<cutoff_ns
        eligible=complete & mature & observed[:,i] & (sign[:,i]!=0)
        summary=conditional_block_pairs(total,late,sign[:,i],eligible,day_ids.astype(np.int64))
        if summary['auc'] is not None:points[i]=summary['auc']
        day_draws[:,i]=bootstrap_conditional_pairs(summary,dw)
        week_draws[:,i]=bootstrap_conditional_pairs(coarsen_conditional_pairs(summary,day_to_week),ww)
        panels.append({'horizon_minutes':i+1,'eligible':summary['eligible_rows'],
            'matched_rows':summary['matched_rows'],'matched_groups':summary['matched_groups'],
            'within_m5_total_up_down_pairs':summary['pair_count'],'conditional_auc':summary['auc'],
            'undefined_day_replicates':int(np.isnan(day_draws[:,i]).sum()),
            'undefined_week_replicates':int(np.isnan(week_draws[:,i]).sum())})
    day_band=conditional_simultaneous_bands(points,day_draws)
    week_band=conditional_simultaneous_bands(points,week_draws)
    for i,panel in enumerate(panels):
        panel['auc_simultaneous_day']=None if day_band is None else {'low':float(day_band['low'][i]),'high':float(day_band['high'][i])}
        panel['auc_simultaneous_week']=None if week_band is None else {'low':float(week_band['low'][i]),'high':float(week_band['high'][i])}
        panel['both_lower_bounds_above_half']=(None if day_band is None or week_band is None
            else bool(day_band['low'][i]>.5 and week_band['low'][i]>.5))
    return {'panels':panels,'input_complete':int(complete.sum()),'parents':n,
        'day_blocks':len(day_values),'week_blocks':len(week_values),'replicates':replicates,
        'day_simultaneous_half_width':None if day_band is None else day_band['half_width'],
        'week_simultaneous_half_width':None if week_band is None else week_band['half_width'],
        'scope':'only exact identical M5 OFI strata; full five-minute M1 input; observed mature nonzero target',
        'limits':'reused development data; selection excludes missing M1, unknown outcomes and ties; does not condition on all other model inputs; no trading or profitability claim'}


STAGE='independent_direction_m1_temporal_conditional_20261029_001'
REPO='/home/andre2/src/GX1_CURRENT'
ROOT='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
SOURCE_PATHS=(
    'gx1/scripts/audit_direction_m1_conditional_v1.py',
    'gx1/contracts/entry_closed_m1_ofi_window_v1.py',
    'gx1/scripts/audit_raw_gc_m1_reference_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/scripts/score_entry_direction_dense_minutes_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/scripts/materialize_entry_direction_dense_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_direction_m1_conditional_v1.py',
)


def require_audit_authority(policy, plan_ref):
    """Require a distinct, unclaimed admission; no historical plan can launch."""
    _require(isinstance(policy,dict) and isinstance(policy.get(STAGE),dict), 'AUTHORITY_MISSING')
    gate=policy[STAGE]
    _require(gate.get('status')=='ADMITTED' and gate.get('plan')==plan_ref
        and gate.get('execution_authorized') is True and gate.get('max_invocations')==1
        and gate.get('claimed_invocations')==0 and gate.get('completed_invocations')==0
        and gate.get('relaunch_allowed') is False, 'ONE_SHOT_AUTHORITY')
    _require(all(policy.get(key) is False for key in ('training_enabled','native_training_enabled',
        'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')), 'GLOBAL_AUTHORITY')


def require_audit_plan(plan):
    """Validate the fixed diagnostic population, statistic and budget."""
    _require(set(plan)=={'schema_version','stage','fold','method','inputs','sources','output_dir','wall_seconds',
        'test_data_used','fit_authorized','model_forward_authorized','economic_edge_claimed','relaunch_allowed'}
        and plan.get('economic_edge_claimed') is False and plan.get('relaunch_allowed') is False
        and plan.get('schema_version')=='gx1_direction_m1_conditional_audit_plan_v1' and plan.get('stage')==STAGE
        and plan.get('test_data_used') is False and plan.get('fit_authorized') is False
        and plan.get('model_forward_authorized') is False and plan.get('wall_seconds')==1200
        and plan.get('fold')=={'prefix_rows':47602,'oof_start_row':47602,'oof_end_row':64890,
            'train_cutoff_ns':1735689600000000000,'oof_cutoff_ns':1743465600000000000}
        and plan.get('method')=={'condition':'exact_physical_float32_model_m5_ofi',
            'score':'last_closed_m1_ofi_contracts','input_eligible':'all_five_observed_same_contract',
            'target_eligible':'observed_nonzero_and_terminal_strictly_before_oof_cutoff',
            'horizons_minutes':list(range(1,61)),'primary_horizon_minutes':5,
            'block_schemes':['UTC_DAY','UTC_WEEK_MONDAY'],'replicates':10000,
            'minimum_blocks':8,'day_seed':20261027,'week_seed':20261028,
            'inference':'existing_97.5_percent_max_deviation_per_scheme_all_60_horizons'}, 'PLAN_SCOPE')
    return plan


def main(argv=None):
    import argparse,os,signal,sys,time
    from pathlib import Path
    import pyarrow.dataset  # Resolve optional Arrow imports before the read guard.
    from gx1.scripts import audit_raw_gc_m1_reference_v1  # Resolve the roll owner before the guard.
    from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','plan-sha256','policy-sha256','source-head'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    _require(os.environ.get('CUDA_VISIBLE_DEVICES')==''
        and not any(name.split('.')[0] in ('torch','sklearn','joblib') for name in sys.modules), 'CPU_ONLY')
    initial=candle.git_state();_require(initial==(args.source_head,'work/gx1-current',''), 'SOURCE_HEAD')
    plan_ref={'path':str(candle.regular(args.plan)),'sha256':args.plan_sha256}
    policy_ref={'path':str(Path(REPO)/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require_audit_plan(plan)
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256'])
        require_audit_authority(policy,plan_ref)
        _require(set(plan['sources'])==set(SOURCE_PATHS), 'SOURCE_SET')
        for name,ref in plan['sources'].items():
            _require(ref['path']==str(Path(REPO)/name), 'SOURCE_PATH')
            binding.require_binding(ref,label='source',verify_file=True)
        inputs=plan['inputs']
        _require(set(inputs)=={'corrected_targets','legacy_axes','m1_source','roll','m5_manifest',
            'full_axis_qualification','model_input_parity'}, 'INPUT_SET')
        payload_names={'corrected_targets','legacy_axes','m1_source','roll'}
        for name,ref in inputs.items():
            binding.require_binding(ref,label='input',verify_file=name not in payload_names)
        manifest=binding.read_bound_json(Path(inputs['m5_manifest']['path']),inputs['m5_manifest']['sha256'])
        layout=binding.read_bound_json(Path(manifest['layout']['path']),manifest['layout']['sha256'])
        source_bindings=binding.read_bound_json(Path(manifest['input_bindings']['path']),manifest['input_bindings']['sha256'])
        prior=binding.read_bound_json(Path(inputs['full_axis_qualification']['path']),inputs['full_axis_qualification']['sha256'])
        parity=binding.read_bound_json(Path(inputs['model_input_parity']['path']),inputs['model_input_parity']['sha256'])
        _require(prior['status']=='PASS_REAL_FULL_AXIS_MODEL_CLOCK_GATHER_ONLY' and prior['test_data_used'] is False
            and parity['status']=='COMPLETED_FIXED_MODEL_INPUT_AUDIT' and parity['test_data_used'] is False
            and prior['qualified_clock_content_sha256']=='c379c735c82427538aaf1b5cfa2aaae89622c470c177dd45c3f51be4eef395ac'
            and prior['source_sha256']==plan['sources']['gx1/contracts/entry_closed_m1_ofi_window_v1.py']['sha256'], 'PRIOR_INPUT_EVIDENCE')
        for name,count in (('TRAIN',47602),('Q1_2025_DEVELOPMENT',17288)):
            row=parity['periods'][name]
            _require(row['parents']==row['exact_model_native_clock_joins']==count
                and all(row[k]==0 for k in ('m1_roll_index_mismatched_cells','model_contract_flag_mismatches',
                    'model_contract_flag_missing','flow_mask_mismatch','flow_mask_triplet_mismatch',
                    'ofi_value_mismatch_given_both_observed','count_mismatch_given_both_observed',
                    'coverage_mismatch_given_both_observed')), 'MODEL_INPUT_PARITY:'+name)
        surface=manifest['surfaces']['M5'];cache_ref=source_bindings['cache_arrays']['M5_ts.npy']['ref']
        payloads={k:inputs[k] for k in ('corrected_targets','legacy_axes','m1_source','roll')}
        payloads.update({'m5_'+k:surface[k] for k in ('available_ns','native_row_indices','features')})
        payloads['m5_cache_times']=cache_ref
        for ref in payloads.values():binding.require_binding({'path':ref['path'],'sha256':ref['sha256']},label='payload',verify_file=False)
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    _require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    out=Path(plan['output_dir'])
    _require(out.parent==Path(ROOT) and out.name=='DIRECTION_M1_TEMPORAL_CONDITIONAL_20261029_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    metadata=[plan_ref['path'],policy_ref['path'],inputs['m5_manifest']['path'],manifest['layout']['path'],
        manifest['input_bindings']['path'],inputs['full_axis_qualification']['path'],inputs['model_input_parity']['path'],inputs['roll']['path']]
    input_paths=[v['path'] for v in payloads.values()]
    sys.addaudithook(candle.make_guard(out,input_paths,metadata,source_paths=[x['path'] for x in plan['sources'].values()]))
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM);old_profile=sys.getprofile()
    def expired(*_unused):raise RuntimeError('DIRECTION_M1_CONDITIONAL_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(candle.no_queries_or_models)
    stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads.values():binding.require_binding({'path':ref['path'],'sha256':ref['sha256']},label='payload',verify_file=True)
            roll_bytes=Path(inputs['roll']['path']).read_bytes()
            with np.load(inputs['m1_source']['path'],allow_pickle=False) as z:
                arrays={k:z[k] for k in ('bar_start_ns','available_ns','contract_index','assigned','ofi_contracts','ofi_valid','receive_clock_excluded','book_event_state_excluded')}
            for x in arrays.values():x.setflags(write=False)
            source=windows.qualify_source(arrays)
            with np.load(inputs['corrected_targets']['path'],allow_pickle=False) as z:
                _require(set(z.files)=={'parent_rows','bar_start_ns','decision_ns','dynamic_entry_m1_row',
                    'terminal_time_ns','terminal_source_row','observed','mid_move','mid_return_bps','sign'}, 'TARGET_MEMBERS')
                parents=z['parent_rows'].copy();decisions=z['decision_ns'].copy();starts=z['bar_start_ns'].copy()
                terminal=z['terminal_time_ns'][47602:64890].copy();observed=z['observed'][47602:64890].copy();sign=z['sign'][47602:64890].copy()
            with np.load(inputs['legacy_axes']['path'],allow_pickle=False) as z:
                _require(all(np.array_equal(z[k],v) for k,v in (('parent_rows',parents),('bar_start_ns',starts),('decision_ns',decisions))), 'EXACT_TARGET_AXES')
            _require(np.array_equal(parents,np.arange(118249,dtype=np.int64)) and np.array_equal(decisions,starts+300_000_000_000)
                and np.all(decisions[47602:64890]>=1735689600000000000) and np.all(decisions[47602:64890]<1743465600000000000), 'EXACT_FOLD')
            parents.setflags(write=False);decisions.setflags(write=False)
            available=np.load(surface['available_ns']['path'],mmap_mode='r');native=np.load(surface['native_row_indices']['path'],mmap_mode='r')
            features=np.load(surface['features']['path'],mmap_mode='r');cache=np.load(cache_ref['path'],mmap_mode='r')
            _require(int(available[-1])<1782864000000000000 and int(cache[-1])<1782864000000000000, 'TEST_BOUNDARY')
            clock=windows.qualify_model_clock(source,parent_rows=parents,decision_ns=decisions,roll_bytes=roll_bytes,
                m5_available_ns=available,m5_native_row_indices=native,m5_features=features,
                m5_field_names=layout['field_names']['M5'],m5_cache_start_ns=cache)
            _require(clock.clock_sha256==prior['qualified_clock_content_sha256']
                and source.source_sha256==prior['qualified_source_content_sha256']
                and clock.roll_sha256==inputs['roll']['sha256'], 'QUALIFIED_SOURCE_CLOCK')
            rows=np.arange(47602,64890,dtype=np.int64);window=windows.gather(source,clock,batch_parent_rows=rows)
            complete=window['ofi_observed'].all(axis=1);_require(int(complete.sum())==17135, 'M1_COMPLETE_COUNT')
            at=np.searchsorted(available,decisions[47602:64890]);_require(np.array_equal(available[at],decisions[47602:64890]), 'M5_JOIN')
            fields=layout['field_names']['M5'];raw=np.asarray(features[at,fields.index('gc_m5_observed_book_ofi_contracts')],dtype=np.float32)
            m5_observed=np.asarray(features[at,fields.index('gc_m5_observed_book_ofi_contracts_observed')])==1
            total_from_m1=window['ofi_contracts'].sum(axis=1).astype(np.float32)
            _require(np.all(m5_observed[complete]) and np.array_equal(raw[complete],total_from_m1[complete]), 'M1_M5_TOTAL_PARITY')
            _require(int((observed[:,4] & (sign[:,4]!=0) & (terminal[:,4]<1743465600000000000)).sum())==17192, 'CORRECTED_FIVE_MINUTE_COUNT')
            result=conditional_multi_horizon(raw,np.ascontiguousarray(window['ofi_contracts'][:,-1]),complete,
                np.ascontiguousarray(decisions[47602:64890]),terminal,observed,sign,cutoff_ns=1743465600000000000,
                replicates=10000,minimum_blocks=8,day_seed=20261027,week_seed=20261028)
            input_snapshot=binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged();control_snapshot.require_unchanged()
        _require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock, 'SOURCE_OR_LOCK_CHANGED')
        result.update(schema_version='gx1_direction_m1_conditional_audit_result_v1',status='COMPLETE',plan=plan_ref,
            source_head=args.source_head,primary_horizon_minutes=5,model_forwards=0,optimizer_steps=0,fit_calls=0,
            test_data_used=False,economic_edge_qualified=False)
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
        sys.setprofile(old_profile);signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':raise SystemExit(main())
