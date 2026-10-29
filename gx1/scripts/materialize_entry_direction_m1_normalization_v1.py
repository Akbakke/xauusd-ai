"""One admitted, TRAIN-only M1 OFI normalizer; no model or trading authority.

The fixed plan identifies already-qualified physical inputs.  Admission is a
separate one-shot gate, checked before any feature payload or claim is opened.
"""
from __future__ import annotations
import hashlib
import numpy as np

REPO='/home/andre2/src/GX1_CURRENT'
ROOT='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
STAGE='independent_direction_m1_train_normalization_20261029_001'
SOURCE_PATHS=(
    'gx1/contracts/entry_closed_m1_ofi_window_v1.py',
    'gx1/contracts/entry_native_direction_m1_ofi_normalization_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/audit_raw_gc_m1_reference_v1.py',
    'gx1/scripts/materialize_entry_direction_m1_normalization_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_materialize_entry_direction_m1_normalization_v1.py',
)
FOLD={'prefix_rows':47602,'train_cutoff_ns':1735689600000000000,'full_axis_rows':118249,
    'axis_end_before_ns':1767225600000000000}
METHOD={'statistic':'existing_observed_closed_m1_ofi_train_prefix_normalization',
    'model_clock':'qualified_existing_m5_contract_mask_and_roll',
    'input_members':['bar_start_ns','available_ns','contract_index','assigned','ofi_contracts',
        'ofi_valid','receive_clock_excluded','book_event_state_excluded'],
    'axes_members':['parent_rows','bar_start_ns','decision_ns'],
    'target_values_used':False,'later_observations_in_fit':False}


def _require(ok, reason):
    if not bool(ok):raise RuntimeError('M1_TRAIN_NORMALIZATION_'+reason)


def require_plan(plan):
    _require(isinstance(plan,dict) and set(plan)=={'schema_version','stage','fold','method','inputs','sources',
        'output_dir','wall_seconds','test_data_used','model_forward_authorized','training_authorized',
        'economic_edge_claimed','relaunch_allowed'}
        and plan.get('schema_version')=='gx1_m1_train_normalization_plan_v1' and plan.get('stage')==STAGE
        and plan.get('fold')==FOLD and plan.get('method')==METHOD and plan.get('wall_seconds')==1200
        and all(plan.get(k) is False for k in ('test_data_used','model_forward_authorized',
            'training_authorized','economic_edge_claimed','relaunch_allowed')), 'PLAN_SCOPE')
    return plan


def require_authority(policy, plan_ref):
    _require(isinstance(policy,dict) and isinstance(policy.get(STAGE),dict), 'AUTHORITY_MISSING')
    gate=policy[STAGE]
    _require(gate.get('status')=='ADMITTED' and gate.get('plan')==plan_ref
        and gate.get('execution_authorized') is True and gate.get('normalization_fit_authorized') is True
        and gate.get('max_invocations')==1 and gate.get('claimed_invocations')==0
        and gate.get('completed_invocations')==0 and gate.get('relaunch_allowed') is False, 'ONE_SHOT_AUTHORITY')
    _require(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')), 'GLOBAL_AUTHORITY')


def fit_prefix(source, clock, *, prefix_rows, cutoff_ns):
    """One fit on the fixed earlier prefix; return the existing owner contract."""
    from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
    from gx1.contracts import entry_native_direction_m1_ofi_normalization_v1 as normalization
    _require(isinstance(source,windows.QualifiedSource) and isinstance(clock,windows.QualifiedClock)
        and clock.roll_sha256 is not None, 'QUALIFIED_CLOCK_REQUIRED')
    value=normalization.fit(source,clock,prefix_rows=prefix_rows,cutoff_ns=cutoff_ns)
    fitted=normalization.require(value)
    observed=windows.gather(source,clock,batch_parent_rows=clock.parent_rows[:prefix_rows])['ofi_observed']
    _require(fitted['model_contract_clock_bound'] is True and fitted['prefix_rows']==prefix_rows
        and fitted['cutoff_ns']==cutoff_ns and fitted['observed_minute_count']==int(observed.sum()), 'FIT_ACCOUNTING')
    return fitted


def main(argv=None):
    import argparse,os,signal,sys,time
    from pathlib import Path
    import pyarrow.dataset  # Resolve optional Arrow imports before installing the read guard.
    from gx1.scripts import audit_raw_gc_m1_reference_v1
    from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
    from gx1.contracts import entry_native_direction_m1_ofi_normalization_v1 as normalization
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
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256']);require_plan(plan)
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);require_authority(policy,plan_ref)
        _require(set(plan['sources'])==set(SOURCE_PATHS), 'SOURCE_SET')
        for name,ref in plan['sources'].items():
            _require(ref['path']==str(Path(REPO)/name), 'SOURCE_PATH')
            binding.require_binding(ref,label='source',verify_file=True)
        inputs=plan['inputs']
        _require(set(inputs)=={'legacy_axes','m1_source','roll','m5_manifest','full_axis_qualification'}, 'INPUT_SET')
        payload_names={'legacy_axes','m1_source','roll'}
        for name,ref in inputs.items():binding.require_binding(ref,label='input',verify_file=name not in payload_names)
        manifest=binding.read_bound_json(Path(inputs['m5_manifest']['path']),inputs['m5_manifest']['sha256'])
        layout=binding.read_bound_json(Path(manifest['layout']['path']),manifest['layout']['sha256'])
        source_bindings=binding.read_bound_json(Path(manifest['input_bindings']['path']),manifest['input_bindings']['sha256'])
        prior=binding.read_bound_json(Path(inputs['full_axis_qualification']['path']),inputs['full_axis_qualification']['sha256'])
        _require(prior.get('status')=='PASS_REAL_FULL_AXIS_MODEL_CLOCK_GATHER_ONLY' and prior.get('test_data_used') is False
            and prior.get('qualified_clock_content_sha256')=='c379c735c82427538aaf1b5cfa2aaae89622c470c177dd45c3f51be4eef395ac'
            and prior.get('source_sha256')==plan['sources']['gx1/contracts/entry_closed_m1_ofi_window_v1.py']['sha256'], 'PRIOR_INPUT_EVIDENCE')
        surface=manifest['surfaces']['M5'];cache_ref=source_bindings['cache_arrays']['M5_ts.npy']['ref']
        payloads={key:inputs[key] for key in payload_names}
        payloads.update({'m5_'+key:surface[key] for key in ('available_ns','native_row_indices','features')})
        payloads['m5_cache_times']=cache_ref
        for ref in payloads.values():binding.require_binding({'path':ref['path'],'sha256':ref['sha256']},label='payload',verify_file=False)
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    _require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    out=Path(plan['output_dir'])
    _require(out.parent==Path(ROOT) and out.name=='DIRECTION_M1_NORMALIZATION_TRAIN_20261029_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    metadata=[plan_ref['path'],policy_ref['path'],inputs['m5_manifest']['path'],manifest['layout']['path'],
        manifest['input_bindings']['path'],inputs['full_axis_qualification']['path'],inputs['roll']['path']]
    sys.addaudithook(candle.make_guard(out,[v['path'] for v in payloads.values()],metadata,
        source_paths=[v['path'] for v in plan['sources'].values()]))
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM);old_profile=sys.getprofile()
    def expired(*_unused):raise RuntimeError('M1_TRAIN_NORMALIZATION_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(candle.no_queries_or_models)
    stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads.values():binding.require_binding({'path':ref['path'],'sha256':ref['sha256']},label='payload',verify_file=True)
            roll_bytes=Path(inputs['roll']['path']).read_bytes()
            with np.load(inputs['m1_source']['path'],allow_pickle=False) as archive:
                _require(set(METHOD['input_members']).issubset(archive.files), 'M1_INPUT_MEMBERS')
                arrays={key:archive[key] for key in METHOD['input_members']}
            for array in arrays.values():array.setflags(write=False)
            source=windows.qualify_source(arrays)
            with np.load(inputs['legacy_axes']['path'],allow_pickle=False) as archive:
                _require(set(METHOD['axes_members']).issubset(archive.files), 'AXIS_MEMBERS')
                parents,starts,decisions=(archive[key].copy() for key in METHOD['axes_members'])
            _require(parents.dtype==starts.dtype==decisions.dtype==np.int64
                and np.array_equal(parents,np.arange(FOLD['full_axis_rows'],dtype=np.int64))
                and np.array_equal(decisions,starts+300_000_000_000) and np.all(np.diff(decisions)>0)
                and int(decisions[-1])<FOLD['axis_end_before_ns']
                and int(np.searchsorted(decisions,FOLD['train_cutoff_ns'],side='left'))==FOLD['prefix_rows'], 'STRICT_TRAIN_AXES')
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
            fitted=fit_prefix(source,clock,prefix_rows=FOLD['prefix_rows'],cutoff_ns=FOLD['train_cutoff_ns'])
            _require(normalization.require(fitted)==fitted, 'OUTPUT_CONTRACT')
            input_snapshot=binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged();control_snapshot.require_unchanged()
        _require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock, 'SOURCE_OR_LOCK_CHANGED')
        encoded=binding.canonical_bytes(fitted);publication._publish_bytes(stage/'NORMALIZATION.json',encoded)
        norm_ref={'path':str(out/'OUTPUT/NORMALIZATION.json'),'sha256':hashlib.sha256(encoded).hexdigest()}
        publish(stage/'RESULT.json',{'schema_version':'gx1_m1_train_normalization_result_v1','status':'COMPLETE',
            'plan':plan_ref,'source_head':args.source_head,'normalization':norm_ref,
            'normalization_contract_sha256':fitted['contract_sha256'],'train_parent_rows':FOLD['prefix_rows'],
            'train_observed_minutes':fitted['observed_minute_count'],'normalization_fit_calls':1,
            'target_values_used':False,'model_forwards':0,'optimizer_steps':0,'test_data_used':False,
            'economic_edge_qualified':False})
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
