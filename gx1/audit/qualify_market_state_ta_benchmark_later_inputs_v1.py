"""One-use physical span coverage for the fixed fold1 development panel.

Reuses the six-surface replay and all 31 old-field checks. The normalization
TRAIN prefix is unchanged. Only axis members are read from the axis archive;
no targets, models, checkpoints, fits or optimizer are used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import numpy as np

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as replay
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.features import htf_features as htf

REPO, RUNS, POLICY = replay.REPO, replay.RUNS, replay.POLICY
STAGE = 'market_state_ta_m5_objective_later_inputs_20261112_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
VIEW = RUNS / 'MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/FEATURE_VIEW.json'
SOURCE_BASE = 'b6459b11936e49eb10b1f63a6df81471bef537d5'
PARENT_RESULT = dict(path=str(RUNS / 'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json'),
                     sha256='8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb')
TA_NORMALIZATION = dict(path=str(RUNS / 'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/TA_INPUT_NORMALIZATION.json'),
                        sha256='02eca0cceec636a39b57cb12d55aa07f458653e289d6c5f57b16811e61438693')
TA_CONTRACT = '6c882086902701d5c9503d0c77c72c8e291ed65c44d065171c2d27afa500c3f6'
WALL_SECONDS = 1800
SOURCES = tuple(dict.fromkeys((*replay.SOURCES,
    'gx1/audit/qualify_market_state_ta_benchmark_later_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'tests/test_market_state_ta_benchmark_later_inputs_v1.py')))
SCOPE = dict(fold_id=1, selected_parent_rows=4096, batch_size=16, selected_batches=256,
    batch_indices=list(overlay.LATER_BATCH_INDICES), score_parent_start=29849, score_rows=17753,
    score_start_ns=overlay.LATER_SCORE_START_NS, score_end_exclusive_ns=overlay.LATER_SCORE_END_NS,
    physical_envelopes=6, producer_calls=6, old_registry_fields=31, new_registry_fields=33,
    target_arrays_decoded=0, normalization_fits=0, model_constructions=0, model_forwards=0,
    checkpoint_reads=0, optimizer_steps=0, test_data_used=False)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('BENCHMARK_LATER_INPUT_' + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 2 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return replay.ref(ROOT / name)


def selected_positions(*, fold_id=1):
    panel = overlay.later_panel(fold_id=fold_id)
    return (panel['score_parent_start'] + np.asarray(panel['batch_indices'], np.int64)[:, None] * 16
            + np.arange(16, dtype=np.int64)).reshape(-1)


def build_plan():
    """Bind metadata/file stats only. Row envelopes are derived after claim."""
    parent = replay.read(PARENT_RESULT)
    need(parent['status'] == 'COMPLETE_INPUT_EXTENSION_ONLY' and parent['source_unchanged'] is True,
         'PARENT_COMPLETE')
    old = replay.read(parent['plan'])
    view_ref = replay.ref(VIEW)
    view = replay.read(view_ref)
    normal = replay.read(replay.OLD_NORM)
    ta = replay.read(TA_NORMALIZATION)
    proof = normal['prefix_qualification']['proof']
    need(view['contract_sha256'] == hashlib.sha256(canonical({k:v for k,v in view.items()
         if k != 'contract_sha256'})).hexdigest()
         and view['prefix'] == proof['prefix'] and view['full_axes'] == proof['full_axes']
         and view['physical_widths'] == dict(signal=317,ctx_cont=132,ctx_cat=1,mtf=255)
         and view['full_ta_feature_profile'] is True and view['dedicated_tf_activity'] is True
         and view['dedicated_m5_side'] is True and view.get('dedicated_non_m5_side',False) is False
         and view['prefix_normalization']['ta_base_normalization_sha256'] == ta['contract_sha256'] == TA_CONTRACT
         and view['anchor_span_input_extension']['parent_prefix_reference'] == replay.OLD_NORM,
         'EXACT_VIEW_AND_TRAIN_NORMALIZATION')
    need(view['full_axes']['rows'] == 118249 and view['prefix']['rows'] == 29574
         and view['prefix']['cutoff_ns'] == 1727654400000000000, 'FOLD1_AXES')
    need(view['anchor_span_input_extension']['overlay_owner'] ==
         replay.ref(REPO/'gx1/contracts/entry_anchor_span_feature_inputs_v1.py'), 'CURRENT_OVERLAY_OWNER')
    plan = {k:deepcopy(old[k]) for k in ('inputs','parameters','caches','local_signal','local_clock_path',
        'local_clock_stat','legacy_names','registry_names','mtf_legacy_columns')}
    need(replay.stat(plan['local_signal']['path']) == plan['local_signal']['stat']
         and replay.stat(plan['local_clock_path']) == plan['local_clock_stat'], 'IMMUTABLE_LOCAL_PARENT')
    payloads=[plan['inputs']['enriched'],*plan['inputs']['native_parts'],
              *(v for lane in plan['caches'].values() for v in lane.values())]
    plan['payload_stats']={v['path']:replay.stat(v['path']) for v in payloads}
    plan['inputs'].update(parent_result=PARENT_RESULT,parent_input_plan=parent['plan'],
                         feature_view=view_ref,ta_normalization=TA_NORMALIZATION)
    plan.update(schema_version='gx1_ta_benchmark_later_input_plan_v1',stage=STAGE,
        source_base_head=SOURCE_BASE,source_owners={p:replay.sha(REPO/p) for p in SOURCES},
        full_axes=deepcopy(view['full_axes']),normalization_prefix=deepcopy(view['prefix']),
        view_sha256=view['contract_sha256'],local_sequence_rows=MODEL_NATIVE_SEQ_LEN,
        window_rows=dict(WINDOW_ROWS),scope=deepcopy(SCOPE),output_root=str(ROOT),
        envelope_rule='contiguous_clock_envelopes_verified_by_native_windows_after_replay',
        execution=dict(module='gx1.audit.qualify_market_state_ta_benchmark_later_inputs_v1',
            runner='scripts/gx1_capped_run.sh',job_class='audit',memory='4G',swap='512M',
            timeout_seconds=WALL_SECONDS,kill_after_seconds=10))
    return plan


def inspect(plan_sha, *, execute):
    plan_ref = dict(path=str(PLAN),sha256=plan_sha)
    plan = replay.read(plan_ref)
    need(plan == build_plan(), 'PLAN_IDENTITY')
    head = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    need(Path.cwd().resolve() == REPO
         and subprocess.check_output(['git','branch','--show-current'],text=True).strip() == 'work/gx1-current'
         and subprocess.check_output(['git','status','--porcelain'],text=True) == ''
         and subprocess.run(['git','merge-base','--is-ancestor',SOURCE_BASE,head]).returncode == 0,
         'CLEAN_SOURCE')
    need(not ROOT.exists() or (ROOT.is_dir() and not ROOT.is_symlink() and not list(ROOT.iterdir())),
         'UNCONSUMED_ROOT')
    policy_ref = replay.ref(POLICY)
    policy = replay.read(policy_ref)
    need(all(policy.get(k) is False for k in
             ('training_enabled','native_training_enabled','native_launch_authorized','full_val_allowed')),
         'TRAINING_CLOSED')
    if execute:
        gate = policy.get(STAGE,{})
        need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
             and gate.get('execution_authorized') is True and gate.get('data_reads_authorized') is True
             and gate.get('training_authorized') is False and gate.get('scope') == SCOPE
             and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('relaunch_allowed') is False, 'EXACT_ADMISSION')
    return plan,plan_ref,policy_ref,head


def envelopes_from_clocks(plan, axes, local_clock, clocks, *, fold_id=1):
    """Conservative bounds only; COMPLETE additionally requires native slices."""
    import pandas as pd
    need(set(axes) == {'parent_rows','bar_start_ns','decision_ns'}, 'AXIS_KEYS')
    for name,a in axes.items():
        need(a.dtype == np.int64 and a.shape == (plan['full_axes']['rows'],)
             and np.all(np.diff(a)>0) and replay.axis_sha(a) == plan['full_axes'][name+'_sha256'],
             'FULL_AXIS:'+name)
    need(np.array_equal(axes['bar_start_ns']+300*10**9,axes['decision_ns']), 'DECISION_CLOCK')
    panel=overlay.later_panel(fold_id=fold_id)
    if fold_id == 2:
        prefix=plan.get('normalization_prefix', {})
        need(prefix.get('rows')==panel['prefix_rows']
             and prefix.get('cutoff_ns')==panel['prefix_cutoff_ns'], 'FOLD2_NORMALIZATION_PREFIX')
    left=panel['score_parent_start']
    score_times=axes['decision_ns'][left:left+panel['score_rows']]
    need(np.all(score_times>=panel['score_start_ns'])
         and np.all(score_times<panel['score_end_exclusive_ns']), 'SCORE_INTERVAL')
    chosen = selected_positions(fold_id=fold_id)
    starts,decisions = axes['bar_start_ns'][chosen],axes['decision_ns'][chosen]
    need(len(chosen)==4096 and np.all(decisions>=panel['score_start_ns'])
         and np.all(decisions<panel['score_end_exclusive_ns']), 'FIXED_SCORE_INTERVAL')
    need(plan['local_sequence_rows']==MODEL_NATIVE_SEQ_LEN and plan['window_rows']==dict(WINDOW_ROWS)
         and set(clocks)==set(replay.TFS), 'CANONICAL_WINDOW_LENGTHS')
    pos = replay.selected_positions(local_clock,starts)
    need(np.all(pos>=MODEL_NATIVE_SEQ_LEN-1), 'LOCAL_HISTORY')
    bounds = dict(local=[int(pos[0])-MODEL_NATIVE_SEQ_LEN+1,int(pos[-1])+1])
    for tf in replay.TFS:
        htf.require_multi_tf_timestamp_grid(clocks[tf],timeframe=tf,context='LATER_INPUT_CLOCK')
        right=np.searchsorted(clocks[tf],decisions-int(htf.MULTI_TF_SHIFT[tf].value),side='right')
        left=right-plan['window_rows'][tf]
        need(np.all(left>=0), 'MTF_HISTORY:'+tf)
        bounds[tf]=[int(left.min()),int(right.max())]
    populations={}
    for lane,(left,right) in bounds.items():
        clock=local_clock if lane=='local' else clocks[lane]
        rows=np.arange(left,right,dtype=np.int64)
        times=clock[rows]
        shift=300*10**9 if lane=='local' else int(htf.MULTI_TF_SHIFT[lane].value)
        need(np.all(np.diff(times)>0) and np.all(times+shift<panel['score_end_exclusive_ns']), 'ENVELOPE_CLOCK:'+lane)
        populations[lane]=dict(selected_unique_row_count=len(rows),
            selected_row_indices_sha256=replay.row_sha(rows,lane),
            available_ns_sha256=replay.axis_sha(times+shift),
            time_min_utc=pd.Timestamp(int(times[0]),tz='UTC').isoformat(),
            time_max_utc=pd.Timestamp(int(times[-1]),tz='UTC').isoformat())
    return bounds,populations,chosen,pos


def read_axes_and_clocks(plan, *, fold_id=1):
    """Decode three axis members and clocks; never a target archive member."""
    import pyarrow.parquet as pq
    end=overlay.later_panel(fold_id=fold_id)['score_end_exclusive_ns']
    source=plan['full_axes']['source']
    need(replay.sha(source['path'])==source['sha256'], 'AXIS_ARCHIVE_SHA')
    with np.load(source['path'],allow_pickle=False) as z:
        axes={}
        for k in ('parent_rows','bar_start_ns','decision_ns'):
            value=z[k]
            need(value.dtype==np.int64, 'AXIS_DTYPE:'+k)
            axes[k]=np.array(value,copy=True,order='C')
    parts=[]
    with Path(plan['local_clock_path']).open('rb') as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=65536,columns=['time'],use_threads=False):
            a=np.asarray(batch.column(0).to_numpy(),dtype='datetime64[ns]').view(np.int64)
            stop=int(np.searchsorted(a,end,side='left'))
            parts.append(a[:stop].copy())
            if stop<len(a):break
    local=np.concatenate(parts)
    need(len(local)>0 and np.all(np.diff(local)>0), 'LOCAL_CLOCK')
    clocks={}
    for tf in replay.TFS:
        item=plan['caches'][tf]['ts']
        need(replay.sha(item['path'])==item['sha256'], 'CLOCK_SHA:'+tf)
        raw=np.load(item['path'],mmap_mode='r',allow_pickle=False)
        need(raw.dtype in (np.dtype('int64'),np.dtype('datetime64[ns]')), 'CLOCK_DTYPE:'+tf)
        a=np.asarray(raw,dtype=np.int64)
        stop=int(np.searchsorted(a,end,side='left'))
        clocks[tf]=np.array(a[:stop],dtype=np.int64,copy=True)
        del raw,a
    for a in (*axes.values(),local,*clocks.values()):a.setflags(write=False)
    return axes,local,clocks


def verify_native_windows(plan, bounds, reports, axes, local_clock, chosen, local_positions):
    """The unchanged slicer checks actual composed 255-column input frames."""
    import pandas as pd
    manifest=replay.read(plan['inputs']['registry_manifest'])
    columns=overlay.column_plan(manifest['feature_names'],replay.FIELDS,htf.MULTI_TF_PER_BAR_FEATURES_V4)
    frames={};row_axes={};local_payload=None
    for lane in ('local',*replay.TFS):
        item=reports[lane]['artifact']
        need(replay.sha(item['path'])==item['sha256'], 'SIDECAR_SHA:'+lane)
        with np.load(item['path'],allow_pickle=False) as z:
            need(set(z.files)=={'row_indices','times_ns','native_bar_positions','anchor_span_bars'},'SIDECAR_KEYS')
            rows,times,spans=(z[k].copy() for k in ('row_indices','times_ns','anchor_span_bars'))
        need(np.array_equal(rows,np.arange(*bounds[lane],dtype=np.int64)), 'ENVELOPE_ROWS:'+lane)
        if lane=='local':
            local_payload=(rows,times)
            continue
        parent=np.load(plan['caches'][lane]['feats']['path'],mmap_mode='r',allow_pickle=False)
        values=overlay.combine_selected(parent,rows,spans,columns=columns)
        del parent
        values.setflags(write=False);times.setflags(write=False);rows.setflags(write=False)
        frame=pd.DataFrame(values,index=pd.DatetimeIndex(times.astype('datetime64[ns]'),tz='UTC'),
                           columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,copy=False)
        frame.attrs.update(ts_int64=times,feats_np=values,causal_warmup_rows=0,
                           htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT)
        frames[lane]=frame;row_axes[lane]=rows
    counts=dict(local_windows=0,native_window_calls=0,mtf_windows=0)
    observed={tf:[None,None] for tf in replay.TFS}
    for index,position in zip(chosen,local_positions):
        ids=np.arange(int(position)-MODEL_NATIVE_SEQ_LEN+1,int(position)+1,dtype=np.int64)
        at=replay.selected_positions(local_payload[0],ids)
        need(np.array_equal(local_payload[1][at],local_clock[ids])
             and np.all(local_clock[ids]+300*10**9<=axes['decision_ns'][index]), 'LOCAL_WINDOW_PARITY')
        counts['local_windows']+=1
        windows=htf.get_model_native_multi_tf_route_windows(frames,
            decision_bar_start=pd.Timestamp(int(axes['bar_start_ns'][index]),tz='UTC'),
            per_tf_seq_lens={tf:plan['window_rows'][tf] for tf in replay.TFS},
            route_timeframes=replay.TFS,base_bar_duration=pd.Timedelta(seconds=300),return_row_indices=True)
        counts['native_window_calls']+=1
        for tf,(values,ids) in windows.items():
            original=row_axes[tf][ids]
            need(values.shape==(plan['window_rows'][tf],255) and np.all(np.diff(original)==1)
                 and np.all(frames[tf].attrs['ts_int64'][ids]+int(htf.MULTI_TF_SHIFT[tf].value)
                            <=axes['decision_ns'][index]), 'NATIVE_WINDOW:'+tf)
            observed[tf][0]=int(original[0]) if observed[tf][0] is None else min(observed[tf][0],int(original[0]))
            observed[tf][1]=int(original[-1])+1
            counts['mtf_windows']+=1
    need(counts==dict(local_windows=4096,native_window_calls=4096,mtf_windows=20480)
         and all(observed[tf]==bounds[tf] for tf in replay.TFS), 'NATIVE_WINDOW_COUNTS_AND_ENVELOPES')
    return counts


def coverage_record(plan, view, axes, populations, result_ref, *, fold_id=1):
    panel=overlay.later_panel(fold_id=fold_id)
    chosen=selected_positions(fold_id=fold_id)
    coverage=dict(schema_version=overlay.LATER_COVERAGE_SCHEMA,status='COMPLETE',qualification=result_ref,
        view_sha256=plan['view_sha256'],full_axes=deepcopy(plan['full_axes']),
        normalization_prefix=deepcopy(plan['normalization_prefix']),
        score_parent_start=panel['score_parent_start'],score_rows=panel['score_rows'],
        score_start_ns=panel['score_start_ns'],score_end_exclusive_ns=panel['score_end_exclusive_ns'],
        batch_size=16,batch_indices=panel['batch_indices'],
        selected_parent_rows_sha256=replay.axis_sha(axes['parent_rows'][chosen]),
        selected_decision_ns_sha256=replay.axis_sha(axes['decision_ns'][chosen]),
        physical_populations=populations)
    overlay._require_later_coverage(view,coverage,axes,populations)
    return coverage


def bookend(plan,plan_ref,policy_ref,head):
    need(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==head
         and subprocess.check_output(['git','status','--porcelain'],text=True)==''
         and replay.sha(PLAN)==plan_ref['sha256'] and replay.sha(POLICY)==policy_ref['sha256']
         and all(replay.sha(REPO/p)==h for p,h in plan['source_owners'].items())
         and all(replay.sha(v['path'])==v['sha256'] for k,v in plan['inputs'].items()
                 if k not in ('native_parts','enriched'))
         and replay.sha(plan['full_axes']['source']['path'])==plan['full_axes']['source']['sha256']
         and all(replay.stat(p)==s for p,s in plan['payload_stats'].items())
         and replay.stat(plan['local_clock_path'])==plan['local_clock_stat']
         and replay.stat(plan['local_signal']['path'])==plan['local_signal']['stat'], 'SOURCE_AND_INPUT_BOOKEND')


def run(plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
    began=time.monotonic();cap=require_capped_cpu_audit_execution()
    plan,plan_ref,policy_ref,head=inspect(plan_sha,execute=True)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.), 'AMBIENT_TIMER')
    old_alarm=signal.getsignal(signal.SIGALRM)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    ROOT.mkdir(exist_ok=True)
    claim=publish('CLAIM.json',dict(plan=plan_ref,policy=policy_ref,source_head=head,cap=cap,scope=SCOPE))
    result=failure=coverage_ref=None;status='FAILED_CONSUMED';source_unchanged=False
    def expired(*_):raise TimeoutError('BENCHMARK_LATER_INPUT_1800_SECONDS')
    try:
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(1,WALL_SECONDS-30-(time.monotonic()-began)))
        axes,local,clocks=read_axes_and_clocks(plan)
        bounds,populations,chosen,positions=envelopes_from_clocks(plan,axes,local,clocks)
        qualification=publish('AXIS_QUALIFICATION.json',dict(plan=plan_ref,ranges=bounds,populations=populations,
            selected_parent_rows_sha256=replay.axis_sha(axes['parent_rows'][chosen]),
            selected_decision_ns_sha256=replay.axis_sha(axes['decision_ns'][chosen]),scope=SCOPE))
        runtime={**plan,'populations':populations,'scope':{**plan['scope'],'ranges':bounds,
                 'history_end_exclusive_ns':overlay.LATER_SCORE_END_NS}}
        del clocks;gc.collect()
        reports=replay.measure(runtime,ranges=bounds,cutoff=overlay.LATER_SCORE_END_NS,output_root=ROOT)
        counts=verify_native_windows(plan,bounds,reports,axes,local,chosen,positions)
        need(set(reports)=={'local',*replay.TFS} and all(x['parity']['fields']==31
             and x['parity']['float32_bit_mismatched_cells']==0 for x in reports.values()), 'SIX_SURFACE_PARITY')
        bookend(plan,plan_ref,policy_ref,head)
        result=publish('RESULT.json',dict(status='COMPLETE',plan=plan_ref,claim=claim,policy=policy_ref,
            source_head=head,source_unchanged=True,scope=SCOPE,counts={**counts,'producer_calls':len(reports),
            'axis_archives':1,'axis_arrays_decoded':3,'sidecar_archive_decodes':6},
            axis_qualification=qualification,surfaces=reports,coverage_path=str(ROOT/'COVERAGE.json'),
            normalization_prefix=plan['normalization_prefix'],normalization_fitted=False,
            old_cache_modified=False,fresh_model_baseline_complete=False,predictive_value_claimed=False,
            test_data_used=False,elapsed_seconds=time.monotonic()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        coverage_ref=publish('COVERAGE.json',coverage_record(plan,replay.read(plan['inputs']['feature_view']),
                                                           axes,populations,result))
        status='COMPLETE'
    except BaseException as exc:
        failure=publish('FAILED.json',dict(type=type(exc).__name__,error=str(exc)[:4096],plan=plan_ref))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        error=None
        try:
            bookend(plan,plan_ref,policy_ref,head);source_unchanged=True
        except BaseException as exc:
            error=exc;status='FAILED_CONSUMED'
            if failure is None:failure=publish('FAILED.json',dict(type=type(exc).__name__,error=str(exc)[:4096],plan=plan_ref))
        terminal=replay.terminal_record(status=status,plan_ref=plan_ref,result=result,failure=failure,
                                       head=head,elapsed=time.monotonic()-began)
        terminal.update(coverage=coverage_ref,source_unchanged=source_unchanged)
        publish('TERMINAL.json',terminal)
        if error is not None:raise RuntimeError('BENCHMARK_LATER_INPUT_BOOKEND') from error


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--build-plan',action='store_true')
    modes.add_argument('--preflight',action='store_true')
    parser.add_argument('--plan-sha256')
    argv=list(sys.argv[1:] if argv is None else argv)
    flags=[x.split('=',1)[0] for x in argv if x.startswith('--')]
    need(len(flags)==len(set(flags)), 'DUPLICATE_CLI_FLAG')
    args=parser.parse_args(argv)
    if args.build_plan:
        need(args.plan_sha256 is None and not PLAN.exists(), 'BUILD_CLI')
        _publish_bytes(PLAN,(json.dumps(build_plan(),sort_keys=True,indent=2)+'\n').encode())
        print(json.dumps(replay.ref(PLAN)))
    else:
        need(type(args.plan_sha256) is str and len(args.plan_sha256)==64
             and all(c in '0123456789abcdef' for c in args.plan_sha256), 'PLAN_SHA')
        if args.preflight:
            plan,plan_ref,_,head=inspect(args.plan_sha256,execute=False)
            print(json.dumps(dict(status='READY_NO_CLAIM_NO_ARRAY_DECODE',plan=plan_ref,source_head=head,
                                  scope=plan['scope'],target_arrays_decoded=0,model_constructions=0)))
        else:run(args.plan_sha256)


if __name__=='__main__':main()
