"""Independent observed Entry target reconstruction and existing GC mask attribution.

Read-only TRAIN/development audit. No target, feature, model or Exit is changed.
References use raw quotes and their own arithmetic; producer calls are separate
mechanical comparisons. Never decode an M1 row group intersecting sealed TEST.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from gx1.scripts import gc_source_disposition_v1 as io
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace

REPO=Path('/home/andre2/src/GX1_CURRENT')
PLAN_REL='configs/research/ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_008.json'
PLAN_ID='ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_008'
ROOT=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/PLAN_ID
NS=10**9
MINUTE=60*NS
DAY=86400*NS

def binding_read(b):
    p=Path(b['path']);io.verify_digest(p,b['sha256'],'AUDIT_INPUT')
    return io.read_json(p) if p.suffix=='.json' else p

def reference_horizon(times, bid, ask, decisions, horizon, session_offset, split_end):
    """Independent quote-index/session arithmetic; no production target helper."""
    sessions=np.unique((times-session_offset)//DAY)
    entry=np.searchsorted(times,decisions)
    clipped=np.minimum(entry,len(times)-1)
    good=(entry<len(times)) & (times[clipped]==decisions)
    session=(decisions-session_offset)//DAY
    position=np.searchsorted(sessions,session)
    exists=(position<len(sessions)) & (position+horizon<len(sessions))
    target_pos=np.minimum(position+horizon,len(sessions)-1)
    candidate=sessions[target_pos]*DAY+session_offset+(decisions-(session*DAY+session_offset))
    end=np.searchsorted(times,candidate)
    ec=np.minimum(end,len(times)-1)
    good &= exists & (end<len(times)) & (((times[ec]-session_offset)//DAY)==sessions[target_pos])
    good &= times[ec]<=split_end
    gross=np.full((len(decisions),2),np.nan)
    gross[good,0]=(bid[ec[good]]-ask[clipped[good]])/ask[clipped[good]]*10000
    gross[good,1]=(bid[clipped[good]]-ask[ec[good]])/bid[clipped[good]]*10000
    elapsed=np.full(len(decisions),np.nan)
    elapsed[good]=(times[ec[good]]-decisions[good])//NS
    return {'valid':good,'entry':clipped,'exit':ec,'target_ns':candidate,'gross':gross,'elapsed':elapsed,'session_supported':exists}

def mask_attribution(table, native):
    """All overlapping rejection causes plus a declared sequential waterfall."""
    minutes=table['bar_start_ns'].to_numpy()
    start=native['m5_bar_start_ns'].to_numpy().astype(np.int64)
    queries=start[:,None]+np.arange(5)[None,:]*MINUTE
    positions=np.searchsorted(minutes,queries)
    clipped=np.minimum(positions,len(minutes)-1)
    present=(positions<len(minutes)) & (minutes[clipped]==queries)
    def values(name):
        a=table[name].fill_null(-1 if name=='contract_index' else 0).to_numpy()[clipped]
        return np.where(present,a,-1 if name=='contract_index' else 0)
    contract=values('contract_index')
    reason={'missing_eligible_M1':~present.all(axis=1),
      'no_contract':(present & (contract<0)).any(axis=1),
      'contract_change_inside_M5':present.all(axis=1) & (contract>=0).all(axis=1) & (contract!=contract[:,:1]).any(axis=1),
      'no_trusted_trade_in_some_M1':(present & (values('trades')==0)).any(axis=1)}
    for field in ['invalid_trade','bad_recv_trade','gap','snapshot_trade','duplicate_trade','unknown']:
        reason[field]=(values(field)>0).any(axis=1)
    order=list(reason)
    surviving=np.ones(len(start),bool);waterfall=[]
    for name in order:
        before=int(surviving.sum());surviving &= ~reason[name]
        waterfall.append({'criterion':name,'remaining':int(surviving.sum()),'exclusive_removed_in_this_order':before-int(surviving.sum())})
    unsigned=np.ones(len(start),bool)
    for name in order:
        if name!='unknown':unsigned &= ~reason[name]
    expected_unsigned=native['m5_volume_valid'].to_numpy()
    expected_signed=native['m5_signed_valid'].to_numpy()
    if not np.array_equal(unsigned,expected_unsigned) or not np.array_equal(surviving,expected_signed):
        raise RuntimeError('AUDIT_M5_MASK_REFERENCE_MISMATCH')
    bit=np.zeros(len(start),np.uint16)
    for i,name in enumerate(order):bit |= reason[name].astype(np.uint16)<<i
    unique,counts=np.unique(bit,return_counts=True)
    combinations=[{'reasons':[name for i,name in enumerate(order) if int(mask)&(1<<i)],'M5_windows':int(n)} for mask,n in zip(unique,counts)]
    vol=table['volume'].to_numpy();unk=table['unknown'].to_numpy()
    return {'native_M5_windows':len(start),'unsigned_valid':int(unsigned.sum()),'signed_valid':int(surviving.sum()),
      'signed_coverage_percent':100*float(surviving.mean()),'known_observed_volume_share':1-float(unk.sum())/int(vol.sum()),
      'unknown_only_after_unsigned_quality':int((unsigned & reason['unknown']).sum()),
      'marginal_rejections_overlap':{name:int(v.sum()) for name,v in reason.items()},'waterfall_order':waterfall,'exact_reason_combinations':combinations,
      'current_masks_directly_require_book_continuity':False,'resets_are_not_a_separate_existing_trade_mask':True,
      'cvd_resets_change_segment_meaning_not_extra_signed_M5_eligibility':True,'quality_requirements_changed':False}

def audit_native(plan, directory):
    from gx1.contracts import entry_observed_market_horizons_v1 as producer
    from gx1.contracts import entry_observed_market_v1 as net_owner
    spec=binding_read(plan['native_design'])
    manifests={split:binding_read(spec['files']['entry_'+split+'_manifest']) for split in ['train','val']}
    source=manifests['train']['extra']['unified_exit_lifecycle']
    source_binding={'path':source['m1_source_path'],'sha256':source['m1_source_sha256']}
    path=binding_read(source_binding);q=pq.ParquetFile(path)
    seal=pd.Timestamp(plan['TEST_seal_utc']).value
    time_column=q.schema_arrow.names.index('time');allowed=[];skipped=[]
    for row in range(q.metadata.num_row_groups):
        stats=q.metadata.row_group(row).column(time_column).statistics
        if stats is None or not stats.has_min_max:raise RuntimeError('AUDIT_RAW_M1_BOUNDARY_UNPROVEN')
        (allowed if pd.Timestamp(stats.max).value<seal else skipped).append(row)
    columns=['time','bid_open','ask_open','bid_high','bid_low','ask_high','ask_low']
    raw=q.read_row_groups(allowed,columns=columns,use_threads=False).to_pandas()
    t=pd.DatetimeIndex(raw.time).as_unit('ns').asi8
    if np.any(t>=seal) or np.any(np.diff(t)<=0):raise RuntimeError('AUDIT_RAW_M1_CLOCK_INVALID')
    price={c:raw[c].to_numpy(np.float64) for c in columns[1:]}
    contract=producer.entry_observed_market_horizons_contract();offset=contract['session_boundary_utc_offset_seconds']*NS
    authority=binding_read(spec['files']['train_cost_authority'])
    par=authority['parameters'];year=net_owner.SECONDS_PER_YEAR
    execution=2*(par['commission']['bps_per_execution']+par['execution_slippage']['central_bps_per_execution'])
    annual=np.array([par['financing_or_swap'][s+'_annual_cost_rate']*10000 for s in ['long','short']])
    results={};examples=[];finite_counts={}
    source_gaps=np.r_[0,np.cumsum(np.diff(t)>MINUTE)]
    for split in ['train','val']:
        file=binding_read(spec['files']['entry_'+split+'_parquet'])
        label_cols=[c for h in contract['observation_horizons_trading_days'] for c in [*contract['gross_target_columns'][str(h)],contract['elapsed_columns'][str(h)]]]
        frame=pq.read_table(file,columns=['time',*label_cols]).to_pandas()
        decisions=pd.DatetimeIndex(frame.time).as_unit('ns').asi8+5*MINUTE
        split_end=pd.Timestamp(plan['split_end_utc'][split]).value
        references={h:reference_horizon(t,price['bid_open'],price['ask_open'],decisions,h,offset,split_end) for h in [1,5,20]}
        # TRAIN ends before the safe source prefix ends. VAL only has its
        # h20-supported prefix reconstructed; no mixed TEST group is decoded.
        checked=np.ones(len(frame),bool) if split=='train' else references[20]['session_supported']
        finite_counts[split]={str(h):int(np.isfinite(frame[contract['gross_target_columns'][str(h)][0]]).sum()) for h in [1,5,20]}
        for h,r in references.items():
            lc,sc=contract['gross_target_columns'][str(h)];ecol=contract['elapsed_columns'][str(h)]
            actual=frame[[lc,sc]].to_numpy(np.float32);elapsed=frame[ecol].to_numpy(np.float32)
            expected=r['gross'].astype(np.float32)
            if not np.array_equal(np.isfinite(actual[checked]),np.isfinite(expected[checked])):
                raise RuntimeError('AUDIT_NATIVE_VALIDITY_PARITY:'+split+':'+str(h))
            valid=checked & r['valid']
            if not np.array_equal(actual[valid],expected[valid]) or not np.array_equal(elapsed[valid],r['elapsed'][valid].astype(np.float32)):
                raise RuntimeError('AUDIT_NATIVE_PRICE_OR_ELAPSED_PARITY:'+split+':'+str(h))
            net=(actual[valid].astype(float)-execution-r['elapsed'][valid,None]*annual[None,:]/year).astype(np.float32)
            production=net_owner.build_entry_observed_market_targets(gross_return_bps=torch.tensor(actual[valid]),elapsed_wall_clock_seconds=torch.tensor(r['elapsed'][valid].astype(np.int64)),economics=plan['native_economics'])['target_bps'].numpy()
            expected_net=np.column_stack((net,np.zeros(len(net),np.float32)))
            if not np.array_equal(production,expected_net):raise RuntimeError('AUDIT_NATIVE_NET_REFERENCE_MISMATCH')
            delay=(t[r['exit'][valid]]-r['target_ns'][valid])//NS
            results[split+'_h'+str(h)]={'rows_reconstructed':int(checked.sum()),'valid':int(valid.sum()),'censored':int((checked & ~r['valid']).sum()),'gross_and_net_float32_bit_parity':True,
                'delayed_target_quote_rows':int((delay>0).sum()),'max_target_quote_delay_seconds':int(delay.max()) if len(delay) else 0,
                'observed_path_has_nonconsecutive_M1_rows':int(np.count_nonzero(source_gaps[r['exit'][valid]]-source_gaps[r['entry'][valid]]))}
            # First example per target argmax and each side of the session
            # boundary, plus the most delayed observed target quote.
            rows=np.flatnonzero(valid);classes=np.argmax(expected_net,axis=1)
            chosen=set()
            for cls in [0,1,2]:
                hit=np.flatnonzero(classes==cls)
                if len(hit):chosen.add(int(rows[hit[0]]))
            clock_mod=decisions[rows]%DAY
            for mod in [offset,(offset-MINUTE*5)%DAY]:
                hit=rows[clock_mod==mod]
                if len(hit):chosen.add(int(hit[0]))
            if len(delay):chosen.add(int(rows[np.argmax(delay)]))
            for row in sorted(chosen):
                i,j=int(r['entry'][row]),int(r['exit'][row]);gross=actual[row].astype(float)
                cost=execution+r['elapsed'][row]*annual/year;target=np.r_[gross-cost,0].astype(np.float32)
                data={'split':split,'native_row':row,'horizon_sessions':h,'M5_bar_open':str(frame.time.iloc[row]),'entry_decision':str(pd.Timestamp(int(t[i]),tz='UTC')),'desired_exit':str(pd.Timestamp(int(r['target_ns'][row]),tz='UTC')),'actual_exit':str(pd.Timestamp(int(t[j]),tz='UTC')),
                  'entry_bid':float(price['bid_open'][i]),'entry_ask':float(price['ask_open'][i]),'exit_bid':float(price['bid_open'][j]),'exit_ask':float(price['ask_open'][j]),'elapsed_seconds':int(r['elapsed'][row]),'long_gross_bps':float(gross[0]),'short_gross_bps':float(gross[1]),'execution_cost_bps':execution,'long_financing_cost_bps':float(cost[0]-execution),'short_financing_cost_bps':float(cost[1]-execution),'long_target_bps':float(target[0]),'short_target_bps':float(target[1]),'flat_target_bps':0.,'argmax':['LONG','SHORT','FLAT'][int(target.argmax())],
                  'observed_long_MFE_bps':float((price['bid_high'][i:j].max()/price['ask_open'][i]-1)*10000),'observed_long_MAE_bps':float((1-price['bid_low'][i:j].min()/price['ask_open'][i])*10000),'observed_short_MFE_bps':float((1-price['ask_low'][i:j].min()/price['bid_open'][i])*10000),'observed_short_MAE_bps':float((price['ask_high'][i:j].max()/price['bid_open'][i]-1)*10000),'path_gap_count':int(source_gaps[j]-source_gaps[i])}
                examples.append(data)
        del frame,references
    io.write_json(directory/'NATIVE_EXAMPLES.json',{'scope':'real quotes, illustrative audit examples; no model/trade selection; observed-path extrema do not fill gaps','rows':examples})
    return {'results':results,'materialized_finite_counts':finite_counts,'source':source_binding,'allowed_original_M1_row_groups':allowed,'skipped_M1_row_groups_intersecting_TEST':skipped,'safe_raw_end_utc':str(pd.Timestamp(int(t[-1]),tz='UTC')),
      'original_TEST_rows_decoded':False,'VAL_tail_raw_reconstruction_completed':False,'exit_policy_used':False,'net_costs':{'round_trip_bps':execution,'annual_cost_bps_long_short':annual.tolist(),'seconds_per_year':year,'favorable_credit_clipped_to_zero':par['financing_or_swap']['favorable_credit_clipped_to_zero']},'economics_are_historical_truth':False}

def audit_research(plan):
    spec=binding_read(plan['GC_probe_plan']);pred_path=binding_read(plan['GC_predictions'])
    frames=[]
    for item in spec['execution_tape_partitions']:
        path=binding_read(item);df=pq.read_table(path,columns=['time','bid_open','ask_open']).to_pandas();frames.append(df)
    frame=pd.concat(frames,ignore_index=True).sort_values('time');t=pd.DatetimeIndex(frame.time).as_unit('ns').asi8
    if np.any(t>=pd.Timestamp(plan['TEST_seal_utc']).value):raise RuntimeError('AUDIT_RESEARCH_TEST_BOUNDARY')
    bid=frame.bid_open.to_numpy(float);ask=frame.ask_open.to_numpy(float)
    pred=pq.read_table(pred_path).to_pandas();results={};examples=[]
    for h,rows in pred.groupby('horizon_minutes',sort=True):
        dec=rows.decision_ns.to_numpy(np.int64);entry=np.searchsorted(t,dec+spec['latency_seconds']*NS)
        end=np.searchsorted(t,t[entry]+int(h)*MINUTE)
        expected=10000*np.log(((bid[end]+ask[end])/2)/((bid[entry]+ask[entry])/2))
        if not np.array_equal(rows.exit_ns.to_numpy(),t[end]) or not np.array_equal(rows.y_mid_bps.to_numpy(),expected):raise RuntimeError('AUDIT_RESEARCH_TARGET_PARITY')
        short_entry=(bid[entry]-ask[end])/bid[entry]*10000
        short_old=(bid[entry]/ask[end]-1)*10000
        diff=short_old-short_entry
        results[str(h)]={'rows':len(rows),'mid_label_float64_bit_parity':True,'short_exit_notional_minus_entry_notional_bps':{'mean':float(diff.mean()),'max':float(diff.max()),'min':float(diff.min())}}
        for q in [int(np.argmax(diff)),int(np.argmin(np.abs(expected)))]:
            examples.append({'horizon_minutes':int(h),'decision':str(pd.Timestamp(int(dec[q]),tz='UTC')),'entry':str(pd.Timestamp(int(t[entry[q]]),tz='UTC')),'exit':str(pd.Timestamp(int(t[end[q]]),tz='UTC')),'entry_bid':float(bid[entry[q]]),'entry_ask':float(ask[entry[q]]),'exit_bid':float(bid[end[q]]),'exit_ask':float(ask[end[q]]),'mid_log_return_bps':float(expected[q]),'short_entry_notional_bps':float(short_entry[q]),'research_short_exit_notional_bps':float(short_old[q]),'difference_bps':float(diff[q])})
    return {'results':results,'examples':examples,'learned_target':'10000*log(exit_mid/entry_mid), not LONG/SHORT/FLAT cash values','exit_policy_used':False,'target_costs_included':False,'strategy_economics_evaluated':False}

def run(binding_path):
    if binding_path!=ROOT/'EXECUTION_BINDING.json':raise RuntimeError('AUDIT_BINDING_PATH')
    binding=io.read_json(binding_path);io.verify_digest(REPO/PLAN_REL,binding['plan_sha256'],'PLAN');plan=io.read_json(REPO/PLAN_REL)
    io.verify_digest(ROOT/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'APPROVAL');approval=io.read_json(ROOT/'OPERATOR_APPROVAL.json')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY');policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['entry_target_and_gc_l1_audit_008']
    if plan['plan_id']!=PLAN_ID or io.git_head()!=binding['source_head'] or gate['status']!='EXECUTION_CLAIMED' or gate['completed_invocations']!=0 or gate['claimed_invocations']!=1 or gate['execution_authorized'] is not False or approval['max_invocations']!=1 or approval['status']!='APPROVED_SINGLE_USE_PENDING_EXECUTION' or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['single_use'] is not True or policy['training_enabled'] is not False:raise RuntimeError('AUDIT_AUTHORITY')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('AUDIT_DIRTY_SOURCE')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('AUDIT_CODE_INVENTORY')
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,'CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    staging=ROOT/'.RESULTS.staging';staging.mkdir(exist_ok=False)
    io.write_json(ROOT/'EXECUTION_CLAIM.json',{'invocation':1,'plan_sha256':binding['plan_sha256'],'source_head':binding['source_head']})
    native=audit_native(plan,staging)
    research=audit_research(plan)
    masks=pq.read_table(binding_read(plan['GC_masks']))
    feature=pq.read_table(binding_read(plan['GC_features']),columns=['m5_bar_start_ns','m5_volume_valid','m5_signed_valid'])
    attribution=mask_attribution(masks,feature)
    result={'status':'AUDIT_COMPLETE_WITH_DEFINITION_AND_COVERAGE_GAPS','source_head':binding['source_head'],'native_targets':native,'research_targets':research,'GC_M5_attribution':attribution,'test_accessed':False,'model_fit_started':False,'target_or_exit_changed':False,'P0_training_started':False,'limits':proof}
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,'CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('AUDIT_SOURCE_CHANGED')
    io.write_json(staging/'RESULT.json',result)
    terminal={'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'test_accessed':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(staging/'RESULT.json')}
    io.write_json(staging/'TERMINAL.json',terminal)
    publish_bundle_directory_noreplace(staging,ROOT/'RESULTS')
    print('ENTRY_AND_L1_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',type=Path,required=True);args=parser.parse_args();run(args.binding)
