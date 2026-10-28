"""Qualify mature direction endpoints and missing M1 timestamps on fixed parents.

No interpolation, model, fitting, checkpoint, broker, sealed TEST or trade.
A complete M1 timestamp sequence does not establish an intrabar tick path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import time
import numpy as np
import pyarrow.parquet as pq

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_observation_gaps_20261028_001'
MINUTE = 60_000_000_000
OFFSET = 5431388
SOURCES = (
    'gx1/scripts/audit_direction_observation_gaps_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_direction_observation_gaps_v1.py',
)


def require(ok, reason):
    if not bool(ok): raise RuntimeError('DIRECTION_OBSERVATION_GAPS_' + reason)


def summarize(decision, terminal, observed, sign, times, cutoff):
    """Count exact endpoint/maturity states, with gaps censored at the cutoff."""
    n = len(decision)
    require(n > 0 and decision.dtype == terminal.dtype == times.dtype == np.int64
        and decision.shape == (n,) and terminal.shape == observed.shape == sign.shape == (n, 60)
        and observed.dtype == np.bool_ and sign.dtype == np.int8 and np.isin(sign, (-1, 0, 1)).all()
        and np.all(np.diff(times) > 0) and np.all(np.diff(times) % MINUTE == 0), 'INPUT')
    require(np.array_equal(terminal, decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE)
        and np.all(decision < cutoff), 'CLOCK')
    entry = np.searchsorted(times, decision)
    require(np.all(entry < len(times)) and np.array_equal(times[entry], decision), 'ENTRY')
    rows = []; gap_cells = {}; unique_gaps = set()
    for j in range(60):
        horizon = j + 1; t = terminal[:, j]; mature = t < cutoff
        ix = np.searchsorted(times, t); within = ix < len(times)
        exact = np.zeros(n, dtype=bool); exact[within] = times[ix[within]] == t[within]
        require(np.array_equal(observed[mature, j], exact[mature]), 'ENDPOINT:' + str(horizon))
        present = mature & observed[:, j]; missing = mature & ~observed[:, j]
        require(np.all(sign[~observed[:, j], j] == 0), 'MISSING_SIGN')
        have = np.flatnonzero(present); delta_rows = ix[have] - entry[have]
        require(np.all((delta_rows >= 0) & (delta_rows <= horizon)), 'SOURCE_STEP_COUNT')
        gaps_in_path = delta_rows < horizon
        absent = np.flatnonzero(missing); right = ix[absent]; left = right - 1
        require(np.all(left >= 0) and np.all(times[left] < t[absent]), 'GAP_LEFT')
        resolved = right < len(times)
        resolved[resolved] &= times[right[resolved]] < cutoff
        censored = int((~resolved).sum())
        for a, b in zip(left[resolved], right[resolved]):
            duration = int((times[b] - times[a]) // MINUTE - 1)
            require(duration > 0, 'GAP_DURATION')
            gap_cells[duration] = gap_cells.get(duration, 0) + 1
            unique_gaps.add((int(a), int(b), duration))
        nonzero = present & (sign[:, j] != 0)
        rows.append({'horizon_minutes':horizon, 'parents':n, 'observed_before_cutoff':int(present.sum()),
            'nonzero_observed_before_cutoff':int(nonzero.sum()), 'zero_observed_before_cutoff':int((present & (sign[:, j] == 0)).sum()),
            'missing_before_cutoff':int(missing.sum()), 'endpoint_at_or_after_cutoff':int((~mature).sum()),
            'missing_right_censored_at_cutoff':censored,
            'observed_with_missing_intermediate_m1':int(gaps_in_path.sum()),
            'nonzero_observed_with_missing_intermediate_m1':int(np.sum(gaps_in_path & (sign[have, j] != 0)))})
    both = (terminal[:, 4] < cutoff) & (terminal[:, 59] < cutoff)
    p5, p60 = observed[:, 4], observed[:, 59]
    support = {'both_horizons_mature':int(both.sum()), 'both_endpoints_observed':int((both & p5 & p60).sum()),
        'five_missing_sixty_observed':int((both & ~p5 & p60).sum()),
        'five_observed_sixty_missing':int((both & p5 & ~p60).sum()),
        'both_endpoints_missing':int((both & ~p5 & ~p60).sum())}
    return {'parents':n, 'per_horizon':rows, 'five_vs_sixty_support':support,
        'missing_cells_by_gap_minutes':{str(k):gap_cells[k] for k in sorted(gap_cells)},
        'distinct_resolved_gaps':len(unique_gaps),
        'distinct_resolved_gaps_by_minutes':{str(k):sum(1 for _,_,d in unique_gaps if d == k)
            for k in sorted({d for _,_,d in unique_gaps})}}


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('design-sha256','policy-sha256','source-head'): parser.add_argument('--'+k,required=True)
    args = parser.parse_args()
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(k.split('.')[0] in ('torch','sklearn','joblib') for k in sys.modules), 'MODEL_PRELOAD')
    initial = candle.git_state(); require(initial == (args.source_head,'work/gx1-current',''), 'SOURCE')
    design_ref = {'path':str(REPO/'docs/DIRECTION_OBSERVATION_GAPS_20261028.json'),'sha256':args.design_sha256}
    policy_ref = {'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        design = binding.read_bound_json(Path(design_ref['path']),design_ref['sha256'])
        policy = binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']); gate = policy[STAGE]
        require(design['schema_version']=='gx1_direction_observation_gaps_design_v1' and design['stage']==STAGE
            and design['test_data_used'] is False and design['cutoff_ns']=={'q1':1743465600000000000,'q2':1751328000000000000}
            and design['native_2025_source_row_offset']==OFFSET, 'DESIGN')
        require(gate['design']==design_ref and gate['status']=='ADMITTED' and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False and all(gate[k] is False for k in ('model_forwards_authorized',
                'optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')), 'GATE')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed',
            'full_val_allowed','native_launch_authorized')), 'TRAINING_CLOSED')
        require(set(design['sources'])==set(SOURCES) and set(design['inputs'])=={'q1','q2','corrected_target','native_2025'}, 'REFERENCES')
        for path, ref in design['sources'].items():
            require(ref['path']==str(REPO/path),'SOURCE_PATH');binding.require_binding(ref,label='source')
        control = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    out = ROOT/'DIRECTION_OBSERVATION_GAPS_20261028_001'
    require(design['output_dir']==str(out) and ROOT.resolve(strict=True)==ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT')
    out.mkdir(exist_ok=False)
    def publish(path,value): publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'design':design_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    sys.addaudithook(candle.make_guard(out,[r['path'] for r in design['inputs'].values()],
        [design_ref['path'],policy_ref['path']],source_paths=[r['path'] for r in design['sources'].values()]))
    stage = out/'.OUTPUT.staging'; start = time.monotonic()
    def expired(*unused): raise RuntimeError('DIRECTION_OBSERVATION_GAPS_WALL_LIMIT')
    previous = signal.signal(signal.SIGALRM,expired); signal.alarm(600)
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in design['inputs'].values(): binding.require_binding(ref,label='input')
            begin = datetime(2025,1,1,tzinfo=timezone.utc); end = datetime(2025,7,1,tzinfo=timezone.utc)
            raw = pq.read_table(design['inputs']['native_2025']['path'],columns=['time'],
                filters=[('time','>=',begin),('time','<',end)])
            times = raw['time'].combine_chunks().to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
            require(len(times)==174253 and np.all(np.diff(times)>0), 'NATIVE_ROWS')
            with np.load(design['inputs']['corrected_target']['path'],allow_pickle=False) as z:
                target={k:z[k].copy() for k in ('parent_rows','decision_ns','dynamic_entry_m1_row','terminal_time_ns',
                    'terminal_source_row','observed','sign')}
            periods={}
            for name in ('q1','q2'):
                with np.load(design['inputs'][name]['path'],allow_pickle=False) as z:
                    archive={k:z[k].copy() for k in ('parent_rows','decision_ns','terminal_time_ns','observed')}
                parents=archive['parent_rows']; ix=np.searchsorted(target['parent_rows'],parents); cutoff=design['cutoff_ns'][name]
                require(len(parents)=={'q1':17288,'q2':1760}[name] and np.all(ix<len(target['parent_rows']))
                    and np.array_equal(target['parent_rows'][ix],parents), 'PARENTS')
                for k in ('decision_ns','terminal_time_ns','observed'): require(np.array_equal(target[k][ix],archive[k]),'AXIS:'+k)
                entry=np.searchsorted(times,archive['decision_ns'])
                require(np.all(entry<len(times)) and np.array_equal(times[entry],archive['decision_ns'])
                    and np.array_equal(entry+OFFSET,target['dynamic_entry_m1_row'][ix]), 'ENTRY_POINTER')
                for j in range(60):
                    end=archive['terminal_time_ns'][:,j]; ok=archive['observed'][:,j] & (end<cutoff)
                    loc=np.searchsorted(times,end[ok]);require(np.all(loc<len(times)) and np.array_equal(times[loc],end[ok])
                        and np.array_equal(loc+OFFSET,target['terminal_source_row'][ix,j][ok]), 'TERMINAL_POINTER')
                periods[name]=summarize(archive['decision_ns'],archive['terminal_time_ns'],archive['observed'],target['sign'][ix],times,cutoff)
            payload=binding.capture_verified_file_snapshot()
        payload.require_unchanged();control.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock,'SOURCE_FREEZE')
        result={'schema_version':'gx1_direction_observation_gaps_result_v1','status':'COMPLETE','design':design_ref,
            'source_head':args.source_head,'inputs':design['inputs'],'native_rows':len(times),'periods':periods,
            'limits':'Missing M1 timestamps are not automatically exchange closures or bad data. Right ends after the fold cutoff are censored. Complete minute timestamps do not prove intrabar ticks, execution, or full path observability.',
            'model_forwards':0,'optimizer_steps':0,'checkpoint_reads':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,
            'source_unchanged':True,'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-start})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT');publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'design':design_ref})
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
