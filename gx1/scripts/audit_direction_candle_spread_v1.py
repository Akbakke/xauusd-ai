"""Describe 1..60 minute target moves relative to recorded candle-open spreads.

The matched bid/ask candle opens are price proxies, not timestamped fills.
This audit grants no model, fit, broker or trading authority.
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
STAGE = 'independent_direction_candle_spread_20261028_001'
OFFSET = 5431388
SOURCES = (
    'gx1/scripts/audit_direction_candle_spread_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_direction_candle_spread_v1.py',
)


def require(ok, reason):
    if not bool(ok): raise RuntimeError('DIRECTION_CANDLE_SPREAD_' + reason)


def describe(bid0, ask0, bid1, ask1, target_sign, target_bps):
    """Compare exact 0.001-price ticks; equal proxy cost is not a win."""
    arrays = [np.asarray(a) for a in (bid0, ask0, bid1, ask1, target_sign, target_bps)]
    require(len({a.shape for a in arrays}) == 1 and arrays[0].ndim == 1 and len(arrays[0]) > 0, 'SHAPE')
    prices = arrays[:4]; sign, bps = arrays[4:]; ticks = []
    for price in prices:
        require(np.isfinite(price).all() and np.all(price > 0), 'PRICES')
        item = np.rint(price * 1000).astype(np.int64)
        require(np.array_equal(item.astype(np.float64) / 1000, price), 'THREE_DECIMAL_GRID')
        ticks.append(item)
    b0, a0, b1, a1 = ticks
    require(np.all(a0 >= b0) and np.all(a1 >= b1), 'CROSSED_QUOTES')
    origin = b0 + a0; delta = b1 + a1 - origin; cost = (a0 - b0) + (a1 - b1)
    require(np.array_equal(np.sign(delta).astype(np.int8), sign)
        and np.allclose(10000 * delta / origin, bps, rtol=1e-12, atol=1e-12), 'TARGET_PARITY')
    nonzero = sign != 0; require(nonzero.any(), 'NONZERO')
    move = np.abs(bps[nonzero]); hurdle = (10000 * cost / origin)[nonzero]
    above = np.abs(delta[nonzero]) > cost[nonzero]; equal = np.abs(delta[nonzero]) == cost[nonzero]
    q = (0.1, 0.25, 0.5, 0.75, 0.9)
    return {'mature_observed':len(sign), 'zero_midpoint_move':int((~nonzero).sum()),
        'eligible_nonzero':int(nonzero.sum()), 'absolute_mid_move_bps_quantiles':np.quantile(move,q).tolist(),
        'entry_candle_spread_bps_quantiles':np.quantile((20000 * (a0-b0)/origin)[nonzero],q).tolist(),
        'two_open_spread_proxy_bps_quantiles':np.quantile(hurdle,q).tolist(),
        'below_one_bp_fraction':float(np.mean(move < 1)),
        'proxy_above_spread_count':int(above.sum()), 'proxy_equal_spread_count':int(equal.sum()),
        'proxy_below_spread_count':int((~above & ~equal).sum()), 'proxy_above_spread_fraction':float(np.mean(above))}


def main():
    sys.dont_write_bytecode = True
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('design-sha256','policy-sha256','source-head'): parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(k.split('.')[0] in ('torch','sklearn','joblib') for k in sys.modules), 'MODEL_PRELOAD')
    initial=candle.git_state();require(initial==(args.source_head,'work/gx1-current',''),'SOURCE')
    design_ref={'path':str(REPO/'docs/DIRECTION_CANDLE_SPREAD_20261028.json'),'sha256':args.design_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        design=binding.read_bound_json(Path(design_ref['path']),design_ref['sha256'])
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);gate=policy[STAGE]
        require(design['schema_version']=='gx1_direction_candle_spread_design_v1' and design['stage']==STAGE
            and design['test_data_used'] is False and design['cutoff_ns']=={'q1':1743465600000000000,'q2':1751328000000000000}
            and design['decimal_places']==3 and design['native_2025_source_row_offset']==OFFSET,'DESIGN')
        require(gate['design']==design_ref and gate['status']=='ADMITTED' and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False and all(gate[k] is False for k in ('model_forwards_authorized',
                'optimizer_steps_authorized','checkpoint_reads_authorized','test_data_used','broker_access_allowed')),'GATE')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed',
            'full_val_allowed','native_launch_authorized')),'TRAINING_CLOSED')
        require(set(design['sources'])==set(SOURCES) and set(design['inputs'])=={'q1','q2','corrected_target','native_2025'},'REFERENCES')
        for path,ref in design['sources'].items():
            require(ref['path']==str(REPO/path),'SOURCE_PATH');binding.require_binding(ref,label='source')
        control=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK')
    out=ROOT/'DIRECTION_CANDLE_SPREAD_20261028_001'
    require(design['output_dir']==str(out) and ROOT.resolve(strict=True)==ROOT and not out.exists() and not out.is_symlink(),'OUTPUT')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'design':design_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    sys.addaudithook(candle.make_guard(out,[r['path'] for r in design['inputs'].values()],
        [design_ref['path'],policy_ref['path']],source_paths=[r['path'] for r in design['sources'].values()]))
    stage=out/'.OUTPUT.staging';start=time.monotonic()
    def expired(*unused):raise RuntimeError('DIRECTION_CANDLE_SPREAD_WALL_LIMIT')
    previous=signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in design['inputs'].values():binding.require_binding(ref,label='input')
            begin=datetime(2025,1,1,tzinfo=timezone.utc);end=datetime(2025,7,1,tzinfo=timezone.utc)
            raw=pq.read_table(design['inputs']['native_2025']['path'],columns=['time','bid_open','ask_open'],
                filters=[('time','>=',begin),('time','<',end)])
            times=raw['time'].combine_chunks().to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
            bid=raw['bid_open'].combine_chunks().to_numpy();ask=raw['ask_open'].combine_chunks().to_numpy()
            require(len(times)==174253 and np.all(np.diff(times)>0) and np.all(ask>=bid),'NATIVE_ROWS')
            with np.load(design['inputs']['corrected_target']['path'],allow_pickle=False) as z:
                target={k:z[k].copy() for k in ('parent_rows','decision_ns','dynamic_entry_m1_row','terminal_time_ns',
                    'terminal_source_row','observed','sign','mid_return_bps')}
            periods={}
            for name in ('q1','q2'):
                with np.load(design['inputs'][name]['path'],allow_pickle=False) as z:
                    archive={k:z[k].copy() for k in ('parent_rows','decision_ns','terminal_time_ns','observed')}
                parents=archive['parent_rows'];ix=np.searchsorted(target['parent_rows'],parents);cutoff=design['cutoff_ns'][name]
                require(len(parents)=={'q1':17288,'q2':1760}[name] and np.all(ix<len(target['parent_rows']))
                    and np.array_equal(target['parent_rows'][ix],parents),'PARENTS')
                for k in ('decision_ns','terminal_time_ns','observed'):require(np.array_equal(target[k][ix],archive[k]),'AXIS:'+k)
                entry=np.searchsorted(times,archive['decision_ns'])
                require(np.all(entry<len(times)) and np.array_equal(times[entry],archive['decision_ns'])
                    and np.array_equal(entry+OFFSET,target['dynamic_entry_m1_row'][ix]),'ENTRY_POINTER')
                panels=[]
                for j in range(60):
                    terminal=archive['terminal_time_ns'][:,j];ok=archive['observed'][:,j]&(terminal<cutoff)
                    loc=np.searchsorted(times,terminal[ok]);require(np.all(loc<len(times)) and np.array_equal(times[loc],terminal[ok])
                        and np.array_equal(loc+OFFSET,target['terminal_source_row'][ix,j][ok]),'TERMINAL_POINTER')
                    row=describe(bid[entry[ok]],ask[entry[ok]],bid[loc],ask[loc],target['sign'][ix,j][ok],target['mid_return_bps'][ix,j][ok])
                    panels.append({'horizon_minutes':j+1,**row})
                periods[name]={'parents':len(parents),'panels':panels}
            payload=binding.capture_verified_file_snapshot()
        payload.require_unchanged();control.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock,'SOURCE_FREEZE')
        result={'schema_version':'gx1_direction_candle_spread_result_v1','status':'COMPLETE','design':design_ref,
            'source_head':args.source_head,'inputs':design['inputs'],'native_rows':len(times),
            'quantile_probabilities':[.1,.25,.5,.75,.9],'periods':periods,
            'limits':'Bid/ask M1 candle opens are price proxies; their exact quote times and execution availability are unknown. No transaction fees, slippage, path, positions or realized returns. No outcome-based selection is authorized.',
            'model_forwards':0,'optimizer_steps':0,'checkpoint_reads':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,
            'source_unchanged':True,'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-start})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT');publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'design':design_ref})
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':main()
