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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from gx1.scripts import gc_source_disposition_v1 as io
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace

REPO=Path('/home/andre2/src/GX1_CURRENT')
PLAN_REL='configs/research/ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_010.json'
PLAN_ID='ENTRY_TARGET_AND_GC_L1_AUDIT_20261017_010'
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

def safe_reference_prefix(reference, source_end_ns, session_offset):
    # A raw row-group boundary can stop in the middle of an otherwise valid
    # target session. It is not proof that later quotes in that session are
    # missing from the actual source. Never score that partial session.
    return (reference['session_supported'] &
            ((reference['target_ns']-session_offset)//DAY <
             (source_end_ns-session_offset)//DAY))

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
        checked=np.ones(len(frame),bool) if split=='train' else safe_reference_prefix(references[20],int(t[-1]),offset)
        finite_counts[split]={str(h):int(np.isfinite(frame[contract['gross_target_columns'][str(h)][0]]).sum()) for h in [1,5,20]}
        for h,r in references.items():
            lc,sc=contract['gross_target_columns'][str(h)];ecol=contract['elapsed_columns'][str(h)]
            actual=frame[[lc,sc]].to_numpy(np.float32);elapsed=frame[ecol].to_numpy(np.float32)
            expected=r['gross'].astype(np.float32)
            if not np.array_equal(np.isfinite(actual[checked]),np.isfinite(expected[checked])):
                bad=np.flatnonzero(checked & (np.isfinite(actual).all(axis=1)!=r['valid']))
                io.write_json(directory/'AUDIT_MISMATCH.json',{'split':split,'horizon':h,'native_rows':bad[:10].tolist(),'decisions':[str(pd.Timestamp(int(decisions[k]),tz='UTC')) for k in bad[:10]],'target_times':[str(pd.Timestamp(int(r['target_ns'][k]),tz='UTC')) for k in bad[:10]],'safe_source_end':str(pd.Timestamp(int(t[-1]),tz='UTC'))})
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
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'POLICY');policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['entry_target_and_gc_l1_audit_010']
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


def evaluate_entry_research(plan, directory):
    from gx1.contracts import entry_research_evaluation_v1 as ev
    spec=binding_read(plan['native_design'])
    manifest=binding_read(spec['files']['entry_train_manifest'])
    source=manifest['extra']['unified_exit_lifecycle']
    path=binding_read({'path':source['m1_source_path'],'sha256':source['m1_source_sha256']})
    q=pq.ParquetFile(path);seal=pd.Timestamp(plan['TEST_seal_utc']).value
    tc=q.schema_arrow.names.index('time');allowed=[]
    for i in range(q.metadata.num_row_groups):
        stats=q.metadata.row_group(i).column(tc).statistics
        if stats is None or not stats.has_min_max:raise RuntimeError('EVAL_RAW_GROUP_BOUNDARY')
        if pd.Timestamp(stats.max).value<seal:allowed.append(i)
    raw=q.read_row_groups(allowed,columns=['time','bid_open','ask_open','bid_high','bid_low','ask_high','ask_low'],use_threads=False).to_pandas()
    t=pd.DatetimeIndex(raw.time).as_unit('ns').asi8
    if np.any(np.diff(t)<=0) or np.any(t>=seal):raise RuntimeError('EVAL_RAW_CLOCK')
    prices={c:raw[c].to_numpy(float) for c in raw if c!='time'}
    calendar=ev.regular_calendar(int(t[0]),int(t[-1]))
    gaps=ev.gap_inventory(t,calendar);gaps.to_parquet(directory/'GAP_CLASSIFICATION.parquet',index=False)
    gap_prefix=np.r_[0,np.cumsum(np.diff(t)>MINUTE)]
    econ=plan['native_economics'];fee=econ['round_trip_execution_cost_bps']
    annual=np.asarray(econ['annual_financing_cost_bps']);year=econ['seconds_per_year']
    authority=binding_read(spec['files']['train_cost_authority'])['parameters']
    expected_fee=2*(authority['commission']['bps_per_execution']+authority['execution_slippage']['central_bps_per_execution'])
    expected_annual=[authority['financing_or_swap'][side+'_annual_cost_rate']*10000 for side in ['long','short']]
    if fee!=expected_fee or not np.array_equal(annual,expected_annual):raise RuntimeError('EVAL_NATIVE_COST_AUTHORITY')
    offset=spec['target_contract']['session_boundary_utc_offset_seconds']*NS
    counts={};example_rows=[];files=[]
    for split in ['train','val']:
        frame=pq.read_table(binding_read(spec['files']['entry_'+split+'_parquet']),columns=['time',*[c for h in [1,5,20] for c in [f'y_long_gross_at_{h}d_bps',f'y_short_gross_at_{h}d_bps',f'elapsed_seconds_at_{h}d']]]).to_pandas()
        decisions=pd.DatetimeIndex(frame.time).as_unit('ns').asi8+5*MINUTE
        end=pd.Timestamp(plan['split_end_utc'][split]).value
        refs={h:reference_horizon(t,prices['bid_open'],prices['ask_open'],decisions,h,offset,end) for h in [1,5,20]}
        checked=np.ones(len(frame),bool) if split=='train' else safe_reference_prefix(refs[20],int(t[-1]),offset)
        for h,r in refs.items():
            actual=frame[[f'y_long_gross_at_{h}d_bps',f'y_short_gross_at_{h}d_bps']].to_numpy(np.float32)
            if not np.array_equal(actual[checked],r['gross'][checked].astype(np.float32),equal_nan=True):raise RuntimeError('EVAL_BASELINE_RAW_PARITY')
            strict,delay=ev.strict_terminal_mask(r,t);valid=checked&r['valid'];strict &= checked
            rows=np.flatnonzero(valid);i=r['entry'][rows];j=r['exit'][rows]
            elapsed=(t[j]-decisions[rows])/NS
            costs=fee+elapsed[:,None]*annual[None,:]/year
            net=(actual[rows].astype(float)-costs).astype(np.float32)
            Lmfe=(ev.range_extrema(prices['bid_high'],i,j,maximum=True)/prices['ask_open'][i]-1)*10000
            Lmae=(1-ev.range_extrema(prices['bid_low'],i,j,maximum=False)/prices['ask_open'][i])*10000
            Smfe=(1-ev.range_extrema(prices['ask_low'],i,j,maximum=False)/prices['bid_open'][i])*10000
            Smae=(ev.range_extrema(prices['ask_high'],i,j,maximum=True)/prices['bid_open'][i]-1)*10000
            candidate=ev.calendar_terminal(t,prices['bid_open'],prices['ask_open'],decisions,h,end,calendar)
            cand_ok=checked&candidate['valid']
            changed=valid&cand_ok&np.any(actual!=candidate['gross'].astype(np.float32),axis=1)
            table=pd.DataFrame({'native_row':rows,'decision_ns':decisions[rows],'baseline_exit_ns':t[j],
              'desired_exit_ns':r['target_ns'][rows],'terminal_delay_seconds':delay[rows],
              'strict_terminal_valid':strict[rows],'calendar_candidate_valid':cand_ok[rows],
              'calendar_candidate_desired_ns':candidate['desired_ns'][rows],
              'baseline_long_net_bps':net[:,0],'baseline_short_net_bps':net[:,1],
              'strict_long_net_bps':np.where(strict[rows],net[:,0],np.nan),
              'strict_short_net_bps':np.where(strict[rows],net[:,1],np.nan),
              'strict_flat_bps':np.where(strict[rows],0.,np.nan),
              'calendar_candidate_long_gross_bps':np.where(cand_ok[rows],candidate['gross'][rows,0],np.nan),
              'calendar_candidate_short_gross_bps':np.where(cand_ok[rows],candidate['gross'][rows,1],np.nan),
              'observed_long_MFE_bps':Lmfe,'observed_long_MAE_bps':Lmae,
              'observed_short_MFE_bps':Smfe,'observed_short_MAE_bps':Smae,
              'expost_long_net_per_adverse_plus_execution':net[:,0]/(Lmae+fee),
              'expost_short_net_per_adverse_plus_execution':net[:,1]/(Smae+fee),
              'interior_missing_M1_intervals':gap_prefix[j]-gap_prefix[i]})
            name=f'{split.upper()}_H{h}_RESEARCH_DIAGNOSTICS.parquet';table.to_parquet(directory/name,index=False);files.append(name)
            # Examples selected by clock order / greatest delay, never a PnL threshold.
            take=sorted(set([0,int(np.argmax(delay[rows]))])) if len(rows) else []
            example_rows.extend([{'split':split,'horizon_sessions':h,**{key:(None if pd.isna(value) else value.item() if isinstance(value,np.generic) else value) for key,value in table.iloc[k].to_dict().items()}} for k in take])
            counts[split+'_h'+str(h)]={'rows_checked':int(checked.sum()),'baseline_valid':int(valid.sum()),
              'strict_valid':int(strict.sum()),'newly_censored_by_exact_terminal':int((valid&~strict).sum()),
              'retained_labels_numerically_changed':0,'baseline_already_censored':int((checked&~r['valid']).sum()),
              'calendar_candidate_valid':int(cand_ok.sum()),'baseline_valid_candidate_censored':int((valid&~cand_ok).sum()),
              'calendar_candidate_changed_among_joint_valid':int(changed.sum()),'calendar_candidate_newly_available':int((cand_ok&~valid).sum()),
              'calendar_candidate_admitted_to_training':False,'max_baseline_delay_seconds':int(delay[valid].max()) if valid.any() else 0,
              'observed_path_extrema_complete_tape_proven':False}
            print('ENTRY_RESEARCH_VARIANT_COMPLETE',split,h,counts[split+'_h'+str(h)]['newly_censored_by_exact_terminal'],flush=True)
        del frame,refs
    io.write_json(directory/'RESEARCH_DIAGNOSTIC_EXAMPLES.json',{'scope':'diagnostics only, no trained model, no hindsight-selected exit','rows':example_rows})
    return {'counts':counts,'strict_variant':ev.STRICT_VARIANT,'max_terminal_delay_seconds':0,
       'calendar_candidate':ev.CALENDAR_CANDIDATE,'historical_calendar_qualified':False,
       'gap_intervals':len(gaps),'gap_missing_minutes':int(gaps.missing_minutes.sum()),
       'missing_minutes_in_current_regular_template':int(gaps.missing_during_current_regular_template.sum()),
       'current_template_closed_missing_minutes':int(gaps.current_template_closed_minutes.sum()),
       'observed_M1_outside_current_template':int((~ev.is_regular_open(t,calendar)).sum()),
       'historical_holiday_vs_source_gap_fully_resolved':False,
       'allowed_original_M1_row_groups':allowed,'safe_raw_end_ns':int(t[-1]),'VAL_tail_reconstructed':False,
       'risk_quality_is_expost_diagnostic_not_feature_or_training_target':True,'Exit_model_used':False,'files':files}


def evaluate_m5_v2(plan):
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    gc=binding_read(plan['GC_probe_plan']);rows=pq.read_table(binding_read(plan['GC_predictions'])).to_pandas()
    parts=[pq.read_table(binding_read(b),columns=['time','bid_open','ask_open']).to_pandas() for b in gc['execution_tape_partitions']]
    df=pd.concat(parts,ignore_index=True).sort_values('time');times=pd.DatetimeIndex(df.time).as_unit('ns').asi8
    if np.any(times>=pd.Timestamp(plan['TEST_seal_utc']).value):raise RuntimeError('EVAL_RESEARCH_TEST')
    bid=df.bid_open.to_numpy(float);ask=df.ask_open.to_numpy(float)
    tape=m5.Tape(times,bid,ask,bid,ask);out={}
    for h,g in rows.groupby('horizon_minutes'):
        dec=g.decision_ns.to_numpy();old=m5.execution_for_horizon(dec,tape,horizon_minutes=int(h),latency_seconds=60)
        new=m5.execution_for_horizon_v2(dec,tape,horizon_minutes=int(h),latency_seconds=60)
        if not new.valid.all() or not np.array_equal(old.y_mid_bps,new.y_mid_bps) or not np.array_equal(new.y_mid_bps,g.y_mid_bps.to_numpy()) or not np.array_equal(new.exit_ns,g.exit_ns.to_numpy()):raise RuntimeError('EVAL_V2_CHANGED_TARGET_OR_CLOCK')
        delta=old.short_gross_bps-new.short_gross_bps
        params=plan['P0_economics'];net={}
        for side in [-1,1]:
            values=np.array([m5.net_trade_bps_v2(side,float(L),float(S),int(n),slippage_bps=params['slippage_bps_per_execution']['central'],financing=params['financing']) for L,S,n in zip(new.long_gross_bps,new.short_gross_bps,new.rollovers)])
            net[str(side)]={'hypothetical_net_mean_bps':float(values.mean()),'rollovers_total':int(new.rollovers.sum())}
        out[str(int(h))]={'rows':len(g),'mid_labels_and_execution_times_unchanged':True,
           'long_v1_v2_max_abs_difference_bps':float(np.max(np.abs(new.long_gross_bps-old.long_gross_bps))),
           'short_v1_minus_v2_mean_bps':float(delta.mean()),'short_v1_minus_v2_max_bps':float(delta.max()),
           'all_opportunity_hypothetical_values_not_strategy_PnL':net}
    return {'accounting_version':m5.EXECUTION_ACCOUNTING_V2,'results':out,
            'same_cost_owner_all_P0_arms':'net_trade_bps_v2','historical_financing_truth_qualified':False,
            'strategy_simulated':False,'published_v1_outputs_changed':False}


def run_evaluation(binding_path):
    identity='ENTRY_EVALUATION_CORRECTION_20261017_011'
    root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    rel='configs/research/'+identity+'.json'
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('EVAL_BINDING_PATH')
    binding=io.read_json(binding_path);io.verify_digest(REPO/rel,binding['plan_sha256'],'EVAL_PLAN');plan=io.read_json(REPO/rel)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'EVAL_POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'EVAL_APPROVAL')
    approval=io.read_json(root/'OPERATOR_APPROVAL.json');policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['entry_evaluation_correction_011']
    if (plan['plan_id']!=identity or gate['status']!='EXECUTION_CLAIMED' or gate['completed_invocations']!=0
       or gate['claimed_invocations']!=1 or gate['execution_authorized'] is not False
       or approval['max_invocations']!=1 or approval['single_use'] is not True
       or approval['approved_plan_sha256']!=binding['plan_sha256'] or policy['training_enabled'] is not False
       or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):
        raise RuntimeError('EVAL_AUTHORITY')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('EVAL_CODE_INVENTORY')
    for path,digest in binding['code_sha256'].items():io.verify_digest(REPO/path,digest,'EVAL_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    staging=root/'.RESULTS.staging';staging.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head'],'plan_sha256':binding['plan_sha256']})
    native=evaluate_entry_research(plan,staging);research=evaluate_m5_v2(plan)
    for path,digest in binding['code_sha256'].items():io.verify_digest(REPO/path,digest,'EVAL_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'EVAL_POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('EVAL_SOURCE_CHANGED')
    result={'status':'RESEARCH_VARIANTS_MEASURED_CALENDAR_UNQUALIFIED','native':native,'GC_M5':research,
      'source_head':binding['source_head'],'limits':proof,'TEST_accessed':False,'fit_started':False,
      'native_baseline_changed':False,'new_targets_trained':False,'production_or_Exit_changed':False}
    io.write_json(staging/'RESULT.json',result)
    io.write_json(staging/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,
      'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(staging/'RESULT.json')})
    publish_bundle_directory_noreplace(staging,root/'RESULTS');print('ENTRY_EVALUATION_CORRECTION_COMPLETE',flush=True)


def l1_quality_worker(items,tick,wanted,expected):
    from gx1.contracts import gc_continuous_source_v1 as owner
    c=owner.L1QualityCollector(wanted,tick)
    old=owner.scan_symbol(items,tick,on_batch=c,extra_columns=owner.L1_AUDIT_COLUMNS,on_file_start=c.file_start)
    for name in owner.METRICS:
        if not np.array_equal(old[name][wanted],expected[name]):raise RuntimeError('L1_EXISTING_TRADE_MASK_METRIC_CHANGED:'+name)
    return {'symbol':old['symbol'],'files':old['files'],'values':c.values,
            'trade_metrics_exactly_reconciled':True}


def build_l1_quality_masks(table,metrics,native):
    from gx1.contracts import gc_continuous_source_v1 as owner
    minute=table['bar_start_ns'].to_numpy();contract=table['contract_index'].fill_null(-1).to_numpy()
    clean=(metrics['records']>0)&(metrics['clean_complete_bbo']>0)
    for key in ['bad_bbo','gap','bad_recv','snapshot','reset','duplicate_trade_signature','file_seam']:
        clean &= metrics[key]==0
    prior=np.r_[False,(np.diff(minute)==MINUTE)&(contract[1:]==contract[:-1])&(contract[1:]>=0)]
    prior &= np.r_[False,clean[:-1]]
    candidate=clean&prior
    # Each family has an explicit column even while its conservative candidate
    # criteria coincide. These are input masks, not admitted feature values.
    q=table.select(['bar_start_ns','contract_index'])
    for name,values in metrics.items():q=q.append_column('L1_'+name,pa.array(values))
    masks={'bbo_snapshot_observed':metrics['clean_complete_bbo']>0,
           'bbo_time_integral_candidate':candidate,'ofi_event_candidate':candidate}
    for key,values in masks.items():q=q.append_column(key+'_valid',pa.array(values))
    start=native['m5_bar_start_ns'].to_numpy();query=start[:,None]+np.arange(5)[None,:]*MINUTE
    pos=np.searchsorted(minute,query);idx=np.minimum(pos,len(minute)-1)
    present=(pos<len(minute))&(minute[idx]==query)
    same=present.all(axis=1)&(contract[idx]>=0).all(axis=1)&(contract[idx]==contract[idx[:,:1]]).all(axis=1)
    out={};months=pd.DatetimeIndex(start,tz='UTC').strftime('%Y-%m').to_numpy()
    for key,v in masks.items():
        full=same & v[idx].all(axis=1)
        out[key]={'valid_M1':int(v.sum()),'valid_native_M5':int(full.sum()),
          'native_M5_by_month':{m:int(np.count_nonzero(full&(months==m))) for m in sorted(set(months))},
          'signed_trade_mask_required':False,'feature_values_built':False,
          'input_candidate_only_not_feature_admission':True}
    return q,out


def run_l1_quality(binding_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import pyarrow as pa
    from gx1.contracts import gc_continuous_source_v1 as owner
    identity='GC_SELECTED_L1_QUALITY_20261017_012'
    root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    rel='configs/research/'+identity+'.json'
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('L1_BINDING_PATH')
    binding=io.read_json(binding_path);io.verify_digest(REPO/rel,binding['plan_sha256'],'L1_PLAN');plan=io.read_json(REPO/rel)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'L1_POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'L1_APPROVAL')
    approval=io.read_json(root/'OPERATOR_APPROVAL.json');policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_selected_l1_quality_012']
    if (plan['plan_id']!=identity or gate['status']!='EXECUTION_CLAIMED' or gate['completed_invocations']!=0
       or gate['claimed_invocations']!=1 or gate['execution_authorized'] is not False
       or approval['max_invocations']!=1 or approval['single_use'] is not True
       or approval['approved_plan_sha256']!=binding['plan_sha256'] or policy['training_enabled'] is not False
       or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):
        raise RuntimeError('L1_AUTHORITY')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('L1_CODE_INVENTORY')
    for path,digest in binding['code_sha256'].items():io.verify_digest(REPO/path,digest,'L1_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    staging=root/'.RESULTS.staging';staging.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head'],'plan_sha256':binding['plan_sha256']})
    source=binding_read(plan['source_admission_plan']);groups,hashes=io._bound_gc_inputs(source)
    table=pq.read_table(binding_read(plan['GC_masks']));roll=binding_read(plan['roll_schedule'])
    native=pq.read_table(binding_read(plan['GC_features']),columns=['m5_bar_start_ns'])
    minute=table['bar_start_ns'].to_numpy();contract=table['contract_index'].fill_null(-1).to_numpy()
    raw_idx=(minute-owner.timing.STUDY_START_NS)//MINUTE
    metrics={key:np.zeros(len(minute),np.uint64) for key in owner.L1_QUALITY_METRICS}
    by_symbol={};futures=[]
    with ProcessPoolExecutor(max_workers=plan['worker_count'],mp_context=mp.get_context('fork')) as pool:
        for group in groups:
            symbol=group['raw_symbol'];ci=roll['contract_order'].index(symbol);rows=np.flatnonzero(contract==ci)
            expected={key:table[key].to_numpy()[rows] for key in owner.METRICS}
            tick=roll['metadata_identity'][symbol][2]
            futures.append((rows,pool.submit(l1_quality_worker,group['items'],tick,raw_idx[rows],expected)))
        for rows,future in futures:
            r=future.result()
            for key,v in r['values'].items():metrics[key][rows]=v
            by_symbol[r['symbol']]={k:v for k,v in r.items() if k!='values'}
    if sum(f['records'] for r in by_symbol.values() for f in r['files'])!=source['expected_source_records']:
        raise RuntimeError('L1_RAW_RECORD_RECONCILIATION')
    q,coverage=build_l1_quality_masks(table,metrics,native)
    output=staging/'SELECTED_M1_L1_CANDIDATE_MASKS.parquet';pq.write_table(q,output,compression='zstd')
    if not q.equals(pq.read_table(output)):raise RuntimeError('L1_READBACK')
    io._rehash_sources(hashes,[])
    for path,digest in binding['code_sha256'].items():io.verify_digest(REPO/path,digest,'L1_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'L1_POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('L1_SOURCE_CHANGED')
    result={'status':'SELECTED_L1_QUALITY_MEASURED_CANDIDATE_MASKS_NOT_FEATURE_ADMISSION',
       'source_head':binding['source_head'],'counts':{k:int(v.sum()) for k,v in metrics.items()},
       'coverage':coverage,'by_symbol':by_symbol,'limits':proof,'whole_feed_completeness_proven':False,
       'reset_or_file_seam_bridged':False,'old_trade_masks_unchanged':True,
       'new_OFI_or_BBO_feature_values_built':False,'TEST_accessed':False,'fit_started':False}
    io.write_json(staging/'RESULT.json',result)
    io.write_json(staging/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
       'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),
       'result_sha256':io.sha256_file(staging/'RESULT.json')})
    publish_bundle_directory_noreplace(staging,root/'RESULTS');print('GC_SELECTED_L1_QUALITY_COMPLETE',flush=True)


def review_cash_projection(plan):
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    gc=binding_read(plan['GC_probe_plan']);rows=pq.read_table(binding_read(plan['GC_predictions'])).to_pandas()
    parts=[pq.read_table(binding_read(b),columns=['time','bid_open','ask_open']).to_pandas() for b in gc['execution_tape_partitions']]
    df=pd.concat(parts,ignore_index=True).sort_values('time');times=pd.DatetimeIndex(df.time).as_unit('ns').asi8
    if np.any(times>=pd.Timestamp(plan['TEST_seal_utc']).value):raise RuntimeError('CASH_REVIEW_TEST')
    bid=df.bid_open.to_numpy(float);ask=df.ask_open.to_numpy(float);tape=m5.Tape(times,bid,ask,bid,ask)
    econ=plan['native_economics'];old=plan['old_M5_economics'];out={}
    authority=binding_read(plan['cost_authority'])['parameters']
    expected_fee=2*(authority['commission']['bps_per_execution']+authority['execution_slippage']['central_bps_per_execution'])
    expected_annual=[authority['financing_or_swap'][side+'_annual_cost_rate']*10000 for side in ['long','short']]
    if econ['round_trip_execution_cost_bps']!=expected_fee or econ['annual_financing_cost_bps']!=expected_annual:raise RuntimeError('CASH_COST_AUTHORITY')
    for h,g in rows.groupby('horizon_minutes'):
        ex=m5.execution_for_horizon_v2(g.decision_ns.to_numpy(),tape,horizon_minutes=int(h),latency_seconds=gc['latency_seconds'])
        if not ex.valid.all() or not np.array_equal(ex.exit_ns,g.exit_ns.to_numpy()) or not np.array_equal(ex.y_mid_bps,g.y_mid_bps.to_numpy()):raise RuntimeError('CASH_CHANGED_CLOCK_OR_DIRECTION_LABEL')
        new=m5.net_cash_values_v3(ex,economics=econ)
        elapsed=(ex.exit_ns-ex.entry_ns)/NS
        expected=np.column_stack((ex.long_gross_bps,ex.short_gross_bps))-econ['round_trip_execution_cost_bps']-elapsed[:,None]*np.asarray(expected_annual)[None,:]/econ['seconds_per_year']
        if not np.array_equal(new[:,:2],expected) or not np.array_equal(new[:,2],np.zeros(len(g))):raise RuntimeError('CASH_NATIVE_COST_REFERENCE')
        before=np.column_stack([np.array([m5.net_trade_bps_v2(side,float(L),float(S),int(n),slippage_bps=old['slippage_bps_per_execution']['central'],financing=old['financing']) for L,S,n in zip(ex.long_gross_bps,ex.short_gross_bps,ex.rollovers)]) for side in [1,-1]])
        diff=new[:,:2]-before
        out[str(int(h))]={'rows':len(g),'cash_projection_native_owner_and_independent_formula_bit_parity':True,
           'entry_exit_and_mid_target_unchanged':True,'rollovers_in_old_model':int(ex.rollovers.sum()),
           'v3_minus_v2_net_mean_bps_LONG_SHORT':diff.mean(axis=0).tolist(),
           'v3_minus_v2_net_min_bps_LONG_SHORT':diff.min(axis=0).tolist(),
           'v3_minus_v2_net_max_bps_LONG_SHORT':diff.max(axis=0).tolist(),
           'opportunity_means_are_not_strategy_PnL':True}
    return {'economics_version':m5.ECONOMICS_PROJECTION_V3,'gross_owner':'execution_for_horizon_v2',
            'cost_owner':'entry_observed_market_v1.build_entry_observed_market_targets',
            'frozen_cost_authority':plan['cost_authority'],'results':out,
            'historical_broker_financing_qualified':False,'same_cash_model_for_all_eight_P0_arms':True,
            'TEST_accessed':False,'strategy_or_fit_run':False,'native_or_published_baseline_changed':False}


def run_cash_review(binding_path):
    identity='GC_M5_CASH_PROJECTION_CHECK_20261017_013';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    rel='configs/research/'+identity+'.json'
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('CASH_BINDING_PATH')
    binding=io.read_json(binding_path);io.verify_digest(REPO/rel,binding['plan_sha256'],'CASH_PLAN');plan=io.read_json(REPO/rel)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'CASH_POLICY');io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'CASH_APPROVAL')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_m5_cash_projection_check_013'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (gate['status']!='EXECUTION_CLAIMED' or gate['completed_invocations']!=0 or gate['claimed_invocations']!=1 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['max_invocations']!=1 or approval['single_use'] is not True or approval['approved_plan_sha256']!=binding['plan_sha256'] or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):
        raise RuntimeError('CASH_AUTHORITY')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('CASH_CODE_INVENTORY')
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'CASH_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry();stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head'],'plan_sha256':binding['plan_sha256']})
    result=review_cash_projection(plan);result['source_head']=binding['source_head'];result['limits']=proof
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'CASH_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'CASH_POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('CASH_SOURCE_CHANGED')
    io.write_json(stage/'RESULT.json',result);io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(stage/'RESULT.json')})
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('CASH_PROJECTION_REVIEW_COMPLETE',flush=True)

def validate_new_fl_inputs_v2(plan,features,levels):
    from gx1.features.gc_order_flow_v1 import L1_FIELDS_V2
    valid=features.ofi_valid.to_numpy()&features.bbo_valid.to_numpy()
    f=features.loc[valid,list(L1_FIELDS_V2)].to_numpy(float)
    if not np.isfinite(f).all() or not np.equal(f[:,:6],np.trunc(f[:,:6])).all() or np.any(f[:,1:6]<0):raise RuntimeError('P0_INPUT_L1_COUNTS')
    if np.any(np.abs(f[:,6])>1) or np.any(f[:,7]<1) or np.any(f[:,8:10]<=0) or np.any(np.abs(f[:,10])>f[:,7]/2+1e-10):raise RuntimeError('P0_INPUT_BBO_PHYSICS')
    q=pq.read_table(binding_read(plan['L1_candidates']),columns=['bar_start_ns','L1_clean_complete_bbo'])
    clock=q['bar_start_ns'].to_numpy();query=features.m5_bar_start_ns.to_numpy()[valid,None]+np.arange(5)[None,:]*MINUTE
    ix=np.searchsorted(clock,query)
    if np.any(ix>=len(clock)) or not np.array_equal(clock[ix],query):raise RuntimeError('P0_INPUT_L1_CLOCK')
    maxevents=int(q['L1_clean_complete_bbo'].to_numpy()[ix].sum(axis=1).max())
    bound=maxevents*2*((1<<32)-2)
    if bound>=2**53:raise RuntimeError('P0_INPUT_OFI_FLOAT_EXACTNESS_NOT_PROVEN')
    cache=pq.read_table(binding_read(plan['full_history_M1']),columns=['bar_start_ns','contract_index','volume_profile_valid','trades','gap','duplicate_trade','bad_recv_trade','snapshot_trade','invalid_trade']).to_pandas()
    clock=cache.bar_start_ns.to_numpy();ci=cache.contract_index.to_numpy();end=(features.decision_ns.to_numpy()-clock[0])//MINUTE
    start=end-levels.session_expected_minutes.to_numpy();inside=(start>=0)&(end<=len(clock));a=np.clip(start,0,len(clock));b=np.clip(end,0,len(clock))
    def seen(v,lo=a):
        prefix=np.r_[0,np.cumsum(np.asarray(v,bool),dtype=np.int64)]
        return (prefix[b]-prefix[lo])>0
    bad=seen(~cache.volume_profile_valid.to_numpy())
    changes=np.r_[False,np.diff(ci)!=0]
    roll=seen(changes,lo=np.minimum(a+1,b))
    direct=inside&~bad&~roll
    if not np.array_equal(direct,levels.session_complete_valid.to_numpy()):raise RuntimeError('P0_INPUT_SESSION_MASK_INDEPENDENT_PREFIX_FAILED')
    reasons={'history_before_raw_scope':~inside,'contract_roll_inside_session':roll,
       'bad_trade_receive_time':seen(cache.bad_recv_trade>0),'invalid_trade':seen(cache.invalid_trade>0),
       'snapshot_trade':seen(cache.snapshot_trade>0),'book_gap_flag':seen(cache.gap>0),
       'repeated_T_signature':seen(cache.duplicate_trade>0),'minute_without_trusted_trade':seen(cache.trades==0)}
    remaining=~direct;waterfall={}
    for name,mask in reasons.items():
        take=remaining&mask;waterfall[name]=int(take.sum());remaining &= ~mask
    if remaining.any():raise RuntimeError('P0_INPUT_SESSION_REJECTION_UNEXPLAINED')
    pub=levels.session_expected_minutes[direct].to_numpy()
    return {'L1_numeric_and_physical_checks':'PASS_ALL_QUALIFIED_ROWS','max_complete_events_per_valid_M5':maxevents,
      'absolute_integer_OFI_accumulation_bound':bound,'float64_exact_integer_limit':2**53,
      'session_mask_vs_independent_full_M1_prefix':'EXACT','session_complete_rows':int(direct.sum()),
      'session_prefix_minutes_median_max':[float(np.median(pub)),int(pub.max())] if len(pub) else None,
      'session_rejection_waterfall_priority_order':waterfall,
      'minute_without_trade_is_not_asserted_to_be_missing_data':True,
      'whole_feed_completeness_proven':False}


def review_managed_money_values_v1(plan):
    legacy=pq.read_table(binding_read(plan['CFTC_legacy'])).to_pandas().sort_values('date')
    dis=pq.read_table(binding_read(plan['CFTC_disaggregated'])).to_pandas().sort_values('date')
    if not np.array_equal(legacy.date,dis.date):raise RuntimeError('P0_INPUT_CFTC_REPORT_CLOCK')
    oi=dis.open_interest_all.to_numpy(float)
    if not np.array_equal(oi,legacy.open_interest_all.to_numpy(float)):raise RuntimeError('P0_INPUT_CFTC_OI_DIFFERENCE')
    checks={}
    for side in ['long','short']:
        name='managed_money_positions_'+side+'_all';change='change_in_managed_money_'+side+'_all'
        delta=dis[name].astype(float).diff();reported=dis[change].astype(float);valid=delta.notna()&reported.notna()
        checks[side]={'compared_reports':int(valid.sum()),'reported_change_vs_consecutive_snapshot_differences':int(np.count_nonzero(delta[valid].to_numpy()!=reported[valid].to_numpy()))}
    mm=(dis.managed_money_positions_long_all.to_numpy(float)-dis.managed_money_positions_short_all.to_numpy(float))/oi
    noncomm=(legacy.non_commercial_positions_long_all.to_numpy(float)-legacy.non_commercial_positions_short_all.to_numpy(float))/oi
    return {'matched_reports':len(dis),'OI_identical_to_legacy':True,'net_over_OI_different_from_legacy_noncommercial_reports':int(np.count_nonzero(mm!=noncomm)),
       'position_change_checks':checks,'current_vintage_only':True,'actual_release_times_and_vintages_qualified':False,'native_features_registered':0}


def qualify_new_fl_p0(plan,stage):
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    from gx1.scripts.research_openbb_source_review_v1 import known_cot_delay_diagnostic_v1
    from gx1.contracts import entry_research_evaluation_v1 as ev
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    from gx1.scripts import research_ta_campaign_v1 as macro
    features=pq.read_table(binding_read(plan['features'])).to_pandas()
    levels=pq.read_table(binding_read(plan['levels'])).to_pandas()
    if not np.array_equal(features.m5_bar_start_ns,levels.m5_bar_start_ns) or not np.array_equal(features.decision_ns,levels.decision_ns):raise RuntimeError('P0_PREP_F_L_CLOCK_MISMATCH')
    selected=features.ofi_valid.to_numpy()|features.bbo_valid.to_numpy()
    if not np.array_equal(features.contract_index.to_numpy()[selected],levels.contract_index.to_numpy()[selected]):raise RuntimeError('P0_PREP_F_L_CONTRACT_MISMATCH')
    input_quality=validate_new_fl_inputs_v2(plan,features,levels)
    macro_comparison=review_managed_money_values_v1(plan)
    for key in levels:
        if key not in ['m5_bar_start_ns','decision_ns','contract_index']:features[key]=levels[key]
    starts=features.m5_bar_start_ns.to_numpy();decisions=features.decision_ns.to_numpy()
    prep=binding_read(plan['preparation']);fields=prep['layers']['F']['fields']+prep['layers']['L']['fields']
    fl=features.ofi_valid.to_numpy()&features.bbo_valid.to_numpy()&features.rolling15_valid.to_numpy()&np.isfinite(features[fields].to_numpy(float)).all(axis=1)
    source=binding_read(plan['execution_plan']);parts=[pq.read_table(binding_read(p),columns=['time','bid_open','ask_open']).to_pandas() for p in source['execution_tape_partitions']]
    data=pd.concat(parts,ignore_index=True).sort_values('time');t=pd.DatetimeIndex(data.time).as_unit('ns').asi8
    if np.any(t>=pd.Timestamp(plan['TEST_seal_utc']).value) or np.any(np.diff(t)<=0) or np.any(t%MINUTE):raise RuntimeError('P0_PREP_TAPE_CLOCK')
    bid=data.bid_open.to_numpy(float);ask=data.ask_open.to_numpy(float)
    if not np.isfinite(bid).all() or not np.isfinite(ask).all() or np.any(bid<=0) or np.any(ask<bid):raise RuntimeError('P0_PREP_TAPE_PRICE')
    tape=m5.Tape(t,bid,ask,bid,ask)
    # Resolve regular weekly closure separately; all residuals stay unclassified.
    take=(t>=starts[0])&(t<decisions[-1]);study=t[take]
    calendar=ev.regular_calendar(int(study[0]),int(study[-1]));gaps=ev.gap_inventory(study,calendar)
    lengths=gaps.missing_minutes.to_numpy(np.int64);count=int(lengths.sum())
    gidx=np.repeat(np.arange(len(gaps)),lengths);offset=np.arange(count)-np.repeat(np.cumsum(lengths)-lengths,lengths)
    missing=gaps.start_ns.to_numpy()[gidx]+offset*MINUTE
    weekly=scheduled_weekly_closure_mask(pd.DatetimeIndex(missing,tz='UTC'))
    gaps.to_parquet(stage/'M5_RESEARCH_GAP_INVENTORY.parquet',index=False)
    spec=io.read_json(REPO/'configs/research/D1_MACRO_CONTEXT_SOURCES_20261011_001.json');cot=macro.parse_cot_gold_archives(spec)
    exceptions=binding_read(plan['COT_exceptions'])
    unsafe=known_cot_delay_diagnostic_v1(decisions,pd.DatetimeIndex(cot.as_of),exceptions['announced_release_utc'])
    out={'input_quality':input_quality,'OpenBB_managed_money_comparison':macro_comparison,'new_F_L_joint_input_rows':int(fl.sum()),'known_old_COT_availability_unsafe_rows':int(unsafe.sum()),
      'COT_comparison_uses_announced_exceptions_not_full_actual_PIT_receipts':True,
      'gap_inventory':{'missing_minutes':count,'weekly_closed_minutes':int(weekly.sum()),'remaining_daily_holiday_or_data_gap_unclassified':int((~weekly).sum()),'current_template_expected_open_missing_minutes':int(gaps.missing_during_current_regular_template.sum()),'historical_daily_holidays_qualified':False},'horizons':{}}
    masks=pd.DataFrame({'decision_ns':decisions,'new_F_L_input_valid':fl,'known_old_COT_unsafe':unsafe})
    for h in prep['horizons_minutes']:
        ex=m5.execution_for_horizon_v2(decisions,tape,horizon_minutes=h,latency_seconds=60)
        strict=m5.exact_intraday_mask_v3(decisions,tape,ex,horizon_minutes=h)
        candidate=fl&strict['valid'];without_known_leak=candidate&~unsafe
        masks['h'+str(h)+'_exact_M1_path']=strict['valid'];masks['h'+str(h)+'_candidate_after_known_COT_exclusion']=without_known_leak
        central=m5.net_cash_values_v3(ex,economics=plan['economics']);sensitivity={}
        elapsed=(ex.exit_ns-ex.entry_ns)/1e9
        expected=np.column_stack((ex.long_gross_bps,ex.short_gross_bps))-plan['economics']['round_trip_execution_cost_bps']-elapsed[:,None]*np.asarray(plan['economics']['annual_financing_cost_bps'])[None,:]/plan['economics']['seconds_per_year']
        if not np.allclose(central[ex.valid,:2],expected[ex.valid],rtol=0,atol=1e-12) or not np.all(central[ex.valid,2]==0):raise RuntimeError('P0_PREP_SHARED_CASH_OWNER')
        # Pure cost deltas from the same v3 formula. No hypothetical strategy PnL.
        for name,scenario in prep['economics_sensitivity'].items():
            cost=scenario['round_trip_execution_cost_bps']+elapsed[:,None]*np.array(scenario['annual_financing_cost_bps'])[None,:]/plan['economics']['seconds_per_year']
            base=plan['economics']['round_trip_execution_cost_bps']+elapsed[:,None]*np.array(plan['economics']['annual_financing_cost_bps'])[None,:]/plan['economics']['seconds_per_year']
            d=base-cost
            sensitivity[name]={'net_delta_vs_central_bps_LONG_SHORT':d[without_known_leak].mean(axis=0).tolist() if without_known_leak.any() else None,'historical_rates_inferred':False}
        out['horizons'][str(h)]={'v2_execution_valid':int(ex.valid.sum()),'exact_M1_path_valid':int(strict['valid'].sum()),'old_valid_with_delayed_terminal':int(np.count_nonzero(ex.valid&(strict['terminal_delay_ns']>0))),'old_valid_with_interior_gap':int(np.count_nonzero(ex.valid&(strict['interior_gap_intervals']>0))),'new_FL_and_exact_path':int(candidate.sum()),'after_known_COT_leak_exclusion':int(without_known_leak.sum()),'same_cost_model_all_eight_arms':True,'eligible_by_month':{m:int(np.count_nonzero(without_known_leak&(pd.DatetimeIndex(decisions,tz='UTC').strftime('%Y-%m').to_numpy()==m))) for m in sorted(set(pd.DatetimeIndex(decisions,tz='UTC').strftime('%Y-%m')))},'cost_sensitivity':sensitivity}
        del central
    masks.to_parquet(stage/'P0_PREPARATION_MASKS.parquet',index=False)
    out.update(models_fitted=False,TEST_accessed=False,P0_training_admitted=False,baseline_Exit_or_native_targets_changed=False,
       blockers=prep['remaining_admission_gates'],native_D1_h20_qualification_separate=True)
    return out


def run_p0_preparation(binding_path):
    identity='GC_P0_COHORT_PREPARATION_20261017_020';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('P0_PREP_BINDING')
    binding=io.read_json(binding_path);rel='configs/research/'+identity+'.json';io.verify_digest(REPO/rel,binding['plan_sha256'],'P0_PREP_PLAN');plan=io.read_json(REPO/rel)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_PREP_POLICY');io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_PREP_APPROVAL')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_P0_cohort_preparation_020'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['max_invocations']!=1 or approval['approved_plan_sha256']!=binding['plan_sha256'] or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):raise RuntimeError('P0_PREP_AUTHORITY')
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_PREP_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry();stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head']})
    result=qualify_new_fl_p0(plan,stage);result['source_head']=binding['source_head'];result['limits']=proof
    for key,value in plan.items():
        if isinstance(value,dict) and set(value)=={'path','sha256'}:binding_read(value)
    execution=binding_read(plan['execution_plan'])
    for value in execution['execution_tape_partitions']:binding_read(value)
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_PREP_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_PREP_POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('P0_PREP_SOURCE_CHANGED')
    io.write_json(stage/'RESULT.json',result);io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(stage/'RESULT.json')})
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('GC_P0_COHORT_PREPARATION_COMPLETE',flush=True)


def p0_matched_prediction_mask(frame,keys):
    """One common prediction metric population, including explicit abstentions.

    Raw per-arm diagnostics are retained. An unseen state is still FLAT for
    strategy execution; predictive family comparisons drop it in every arm.
    """
    return frame.outcome_within_fold.to_numpy() & np.isfinite(frame[keys].to_numpy(float)).all(axis=1)


def run_p0_review(binding_path):
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    from gx1.contracts.entry_observed_market_v1 import build_entry_observed_market_targets
    import hashlib,os
    identity='GC_P0_DEVELOPMENT_REVIEW_20261017_022';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('P0_REVIEW_BINDING')
    binding=io.read_json(binding_path);planpath=REPO/'configs/research'/f'{identity}.json'
    io.verify_digest(planpath,binding['plan_sha256'],'P0_REVIEW_PLAN');plan=io.read_json(planpath)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_REVIEW_POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_REVIEW_APPROVAL')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_p0_review_022'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0
        or gate['execution_authorized'] is not False or policy['training_enabled'] is not False
        or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['single_use'] is not True or approval['max_invocations']!=1):raise RuntimeError('P0_REVIEW_AUTHORITY')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('P0_REVIEW_SOURCE')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('P0_REVIEW_CODE_INVENTORY')
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_REVIEW_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'binding_sha256':io.sha256_file(binding_path)})
    stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False)
    inputs=plan['inputs']
    for item in inputs.values():binding_read(item)
    result=io.read_json(Path(inputs['result']['path']));terminal=io.read_json(Path(inputs['terminal']['path']))
    if terminal['status']!='COMPLETE' or not terminal['source_unchanged'] or terminal['TEST_accessed'] or terminal['result_sha256']!=inputs['result']['sha256']:raise RuntimeError('P0_REVIEW_TERMINAL')
    recipe=io.read_json(Path(inputs['recipe']['path']));source=io.read_json(Path(inputs['native_source_plan']['path']))
    for b in source['execution_tape_partitions']:binding_read(b)
    tape=m5.load_tape(source['reference'],2024)
    if tape.time_ns.max()>=pd.Timestamp(recipe['TEST_seal_utc']).value:raise RuntimeError('P0_REVIEW_TEST_SEAL')
    predictions=pd.read_parquet(inputs['predictions']['path']);trades=pd.read_parquet(inputs['trades']['path']);daily=pd.read_parquet(inputs['daily']['path'])
    # Reconstruct every actually selected side from raw historical quotes.
    entry=np.searchsorted(tape.time_ns,trades.entry_ns);end=np.searchsorted(tape.time_ns,trades.mark_or_exit_ns)
    if not np.array_equal(tape.time_ns[entry],trades.entry_ns) or not np.array_equal(tape.time_ns[end],trades.mark_or_exit_ns):raise RuntimeError('P0_RAW_QUOTE_TIME')
    buy=tape.ask_open[entry];sell=tape.bid_open[entry];exit_bid=tape.bid_open[end];exit_ask=tape.ask_open[end]
    lg=1e4*(exit_bid-buy)/buy;sg=1e4*(sell-exit_ask)/sell;sides=trades.side.to_numpy()
    gross=np.where(sides>0,lg,sg)
    ep=np.where(sides>0,buy,sell);xp=np.where(sides>0,exit_bid,exit_ask)
    if not np.array_equal(ep,trades.entry_price) or not np.array_equal(xp,trades.exit_or_mark_price):raise RuntimeError('P0_RAW_QUOTE_PRICES')
    if not np.allclose(gross,trades.gross_bps,rtol=0,atol=1e-10):raise RuntimeError('P0_RAW_GROSS_NOTIONAL')
    elapsed=((trades.mark_or_exit_ns-trades.entry_ns)//10**9).to_numpy()
    if not np.array_equal(elapsed,trades.elapsed_seconds):raise RuntimeError('P0_FUNDING_ELAPSED')
    cash_checks={};max_error=0.
    # Cost scenarios change the v3 contract hash; no historical authority implied.
    for scenario,config in recipe['cost_scenarios'].items():
        econ=dict(recipe['economics']);econ.update({k:config[k] for k in ['round_trip_execution_cost_bps','annual_financing_cost_bps']})
        payload={k:v for k,v in econ.items() if k!='economics_sha256'}
        econ['economics_sha256']=hashlib.sha256(__import__('json').dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        actual=build_entry_observed_market_targets(gross_return_bps=torch.tensor(np.column_stack((lg,sg)),dtype=torch.float64),elapsed_wall_clock_seconds=torch.tensor(elapsed,dtype=torch.int64),economics=econ)['target_bps'].numpy()
        values=actual[np.arange(len(trades)),np.where(sides>0,0,1)]
        checks={}
        for key,cell in result['cells'].items():
            expected=float(values[trades.strategy.to_numpy()==key].sum())
            error=abs(expected-cell['economics'][scenario]['net_total_bps']);max_error=max(max_error,error)
            if error>1e-8:raise RuntimeError('P0_SHARED_V3_CASH_MISMATCH:'+key)
            checks[key]=expected
        cash_checks[scenario]=checks
    # Decision cost uses the last closed quote, never execution/future spread.
    d=predictions.decision_ns.to_numpy();i=np.searchsorted(tape.time_ns,d-60*10**9,side='right')-1
    spread=(tape.ask_close[i]-tape.bid_close[i])/((tape.ask_close[i]+tape.bid_close[i])/2)*10000
    if not np.allclose(spread,predictions.spread_at_decision_bps,rtol=0,atol=1e-12):raise RuntimeError('P0_DECISION_SPREAD')
    matching={};contrasts={};activity={};coreference={}
    for model in ('rule','tree'):
        keys=[model+'_'+arm for arm in m5.P0_ARMS];good=p0_matched_prediction_mask(predictions,keys);part=predictions[good]
        matching[model]={'rows':int(good.sum()),'excluded_for_unseen_state_or_fold_boundary':int((~good).sum()),'cells':{key:m5.p0_prediction_quality(part.y30_mid_bps.to_numpy(),part[key].to_numpy()) for key in keys}}
        coreference[model]=float(np.mean((part.y30_mid_bps-part.train_mean_bps)**2))
        for key in keys:
            p=predictions[key].to_numpy();a=m5.p0_actions(p,spread,recipe['economics'],30)
            if not np.array_equal(a,predictions['action_'+key]):raise RuntimeError('P0_DECISION_RECONSTRUCTION')
            frame=predictions.assign(_action=a)
            activity[key]={str(f):{'opportunities':len(g),'LONG_signals':int((g._action==1).sum()),'SHORT_signals':int((g._action==-1).sum()),'FLAT_signals':int((g._action==0).sum()),'max_abs_forecast_bps':float(np.nanmax(np.abs(g[key]))),'median_XAU_spread_bps':float(g.spread_at_decision_bps.median())} for f,g in frame.groupby('fold')}
            if key!=model+'_CORE':
                loss=(part.y30_mid_bps-part[model+'_CORE'])**2-(part.y30_mid_bps-part[key])**2
                records=[{'entry_ns':int(t),'improvement':float(v)} for t,v in zip(part.decision_ns,loss)]
                contrasts[key]=m5.weekly_block_bootstrap(records,'improvement',reps=recipe['bootstrap_reps'],seed=recipe['bootstrap_seed'],levels=recipe['confidence_levels'])
    for fold,group in pd.DataFrame(result['fit_receipts']).groupby('fold'):
        if len(group)!=8 or group.train_decision_sha256.nunique()!=1 or group.train_rows.nunique()!=1 or group.eval_rows.nunique()!=1:raise RuntimeError('P0_TRAIN_EVAL_ARM_MATCH')
    months=trades.assign(month=pd.to_datetime(trades.entry_ns,utc=True).dt.strftime('%Y-%m')).groupby(['strategy','month']).size()
    observed={key:{str(m):int(n) for (k,m),n in months.items() if k==key} for key in result['cells']}
    review={'status':'P0_INDEPENDENT_CASH_AND_MATCHED_PREDICTION_REVIEW_COMPLETE','source_head':binding['source_head'],
        'original_RESULT_preserved':True,'raw_quote_reconstructed_trade_or_open_mark_count':len(trades),'maximum_v3_cash_error_bps':max_error,
        'cash_all_scenarios':cash_checks,'matched_prediction':matching,'matched_prediction_contrasts':contrasts,'matched_train_mean_MSE_bps2':coreference,
        'decision_reconstruction':'EXACT_ALL_RULE_TREE_ROWS','train_eval_population_match':'EXACT_EIGHT_ARMS_EACH_FOLD',
        'activity_by_fold':activity,'trade_entries_by_month':observed,'all_selected_tree_positions_LONG':bool((trades.loc[trades.strategy.str.startswith('tree_'),'side']==1).all()),
        'calendar_Sharpe_limit':'Missing quote days forward-fill executable marks; accrued funding is recognized at next observed mark. Observed M1 drawdown cannot bound intra-gap/intraminute loss.',
        'source_unchanged':True,'TEST_accessed':False,'models_refitted':0,'limits':proof}
    for item in inputs.values():binding_read(item)
    for b in source['execution_tape_partitions']:binding_read(b)
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_REVIEW_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_REVIEW_POLICY_POST')
    if io.git_head()!=binding['source_head']:raise RuntimeError('P0_REVIEW_HEAD_POST')
    io.write_json(stage/'RESULT.json',review);io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'result_sha256':io.sha256_file(stage/'RESULT.json'),'source_unchanged':True,'test_data_used':False,'TEST_accessed':False,'models_refitted':0,'finished_utc':datetime.now(timezone.utc).isoformat()})
    for p in stage.iterdir():
        with p.open('rb') as f:os.fsync(f.fileno())
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('P0_REVIEW_COMPLETE',flush=True)


def p0_forecast_diagnostics(y,p,reference,quantiles):
    """Descriptive algebra only, never a fitted calibration or new prediction."""
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    y=np.asarray(y,float);p=np.asarray(p,float);c=np.asarray(reference,float)
    ok=np.isfinite(y)&np.isfinite(p)&np.isfinite(c);y=y[ok];p=p[ok];c=c[ok]
    if not len(y):return {'rows':0}
    ey=float(y.mean());ep=float(p.mean());vp=float(np.var(p));vy=float(np.var(y));cov=float(np.mean((y-ey)*(p-ep)))
    mse=float(np.mean((y-p)**2));ref=float(np.mean((y-c)**2));variance_penalty=float(np.mean((p-c)**2));alignment=float(2*np.mean((y-c)*(p-c)))
    if not np.isclose(mse-ref,variance_penalty-alignment,rtol=0,atol=1e-10):raise RuntimeError('P0_DIAG_MSE_IDENTITY')
    out=m5.p0_prediction_quality(y,p)
    out.update(mean_error_bps=ep-ey,prediction_std_bps=float(np.sqrt(vp)),outcome_std_bps=float(np.sqrt(vy)),
        prediction_positive_share=float(np.mean(p>0)),outcome_positive_share=float(np.mean(y>0)),
        prediction_quantiles_bps={str(q):float(v) for q,v in zip(quantiles,np.quantile(p,quantiles))},
        outcome_quantiles_bps={str(q):float(v) for q,v in zip(quantiles,np.quantile(y,quantiles))},
        forecast_outcome_covariance_bps2=cov,reference_MSE_bps2=ref,MSE_excess_bps2=mse-ref,
        reference_relative_prediction_energy_bps2=variance_penalty,twice_reference_relative_alignment_bps2=alignment,
        MSE_variance_component_bps2=vy+vp-2*cov,MSE_squared_bias_bps2=(ey-ep)**2)
    return out


def p0_fixed_horizon_orders(decisions,sides,execution):
    """Predetermined diagnostic horizon. Flat at decision, one open position.

    No threshold/forecast change, no future-return selection. The source must
    prove exact execution/terminal quotes for every opportunity beforehand.
    Scheduled EXIT is known when entry occurs, never an optimal future exit.
    """
    if not np.all(execution.valid):raise RuntimeError('P0_FIXED_HORIZON_CENSORED_SOURCE')
    selected=[];busy=-1
    for r,side in enumerate(sides):
        if side and decisions[r]>=busy:
            selected.append(r);busy=int(execution.exit_ns[r])
    ix=np.asarray(selected,np.int64)
    times=np.r_[np.asarray(decisions)[ix],execution.exit_ns[ix]-60*10**9]
    actions=np.r_[np.asarray(sides)[ix],np.zeros(len(ix),np.int8)]
    order=np.argsort(times,kind='stable');times=times[order];actions=actions[order]
    if np.any(np.diff(times)<=0):raise RuntimeError('P0_FIXED_HORIZON_ORDER_COLLISION')
    return times,actions,ix


def p0_group_forecasts(frame,keys,quantiles):
    groups={'pooled':np.ones(len(frame),bool)}
    for field in ('fold','session','RV_tercile','EMA_trend_sign'):
        for value in sorted(frame[field].unique()):groups[field+'='+str(value)]=(frame[field].to_numpy()==value)
    for (fold,session,rv),rows in frame.groupby(['fold','session','RV_tercile'],observed=True).groups.items():
        mask=np.zeros(len(frame),bool);mask[np.asarray(rows)]=True;groups[f'fold_session_RV={fold}/{session}/{rv}']=mask
    return {name:{key:p0_forecast_diagnostics(frame.y30_mid_bps.to_numpy()[mask],frame[key].to_numpy()[mask],frame.train_mean_bps.to_numpy()[mask],quantiles) for key in keys} for name,mask in groups.items()}


def run_p0_diagnosis(binding_path):
    import os,hashlib,json,time
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    from gx1.time.session_detector import SESSION_NAME_BY_ID
    identity='GC_P0_CLOSING_DIAGNOSIS_20261017_023';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('P0_DIAG_BINDING_PATH')
    binding=io.read_json(binding_path);planpath=REPO/'configs/research'/f'{identity}.json'
    io.verify_digest(planpath,binding['plan_sha256'],'P0_DIAG_PLAN');plan=io.read_json(planpath)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_DIAG_POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_DIAG_APPROVAL')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_p0_closing_diagnosis_023'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (policy['training_enabled'] is not False or gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0
        or gate['execution_authorized'] is not False or gate['relaunch_authorized'] is not False or gate['model_fit_authorized'] is not False
        or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['single_use'] is not True or approval['max_invocations']!=1):raise RuntimeError('P0_DIAG_SCOPE')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('P0_DIAG_CLEAN_SOURCE')
    if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!='work/gx1-current':raise RuntimeError('P0_DIAG_BRANCH')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError('P0_DIAG_CODE_INVENTORY')
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_DIAG_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'binding_sha256':io.sha256_file(binding_path)})
    stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False);started=time.monotonic()
    try:
        for item in plan['inputs'].values():binding_read(item)
        inputs=plan['inputs'];original=io.read_json(Path(inputs['result']['path']));terminal=io.read_json(Path(inputs['terminal']['path']))
        if terminal['status']!='COMPLETE' or terminal['TEST_accessed'] or not terminal['source_unchanged'] or terminal['result_sha256']!=inputs['result']['sha256']:raise RuntimeError('P0_DIAG_ORIGINAL_TERMINAL')
        recipe=io.read_json(Path(inputs['recipe']['path']));source=io.read_json(Path(inputs['source_plan']['path']))
        frame=pd.read_parquet(inputs['predictions']['path']);old_trades=pd.read_parquet(inputs['trades']['path']);old_daily=pd.read_parquet(inputs['daily']['path'])
        for b in source['execution_tape_partitions']:binding_read(b)
        tape=m5.load_tape(source['reference'],2024)
        if tape.time_ns.max()>=pd.Timestamp(recipe['TEST_seal_utc']).value:raise RuntimeError('P0_DIAG_TEST_SEAL')
        decisions=frame.decision_ns.to_numpy()
        ex=m5.execution_for_horizon_v2(decisions,tape,horizon_minutes=30,latency_seconds=60)
        strict=m5.exact_intraday_mask_v3(decisions,tape,ex,horizon_minutes=30)['valid']
        if not strict.all() or not np.array_equal(ex.y_mid_bps,frame.y30_mid_bps):raise RuntimeError('P0_DIAG_RAW_LABEL_RECONSTRUCTION')
        # Reuse native scalar source, never feature generation, model inference or fitting.
        binding_read(source['feature_inputs']['parquet'])
        base=pq.read_table(source['feature_inputs']['parquet']['path'])
        times,snap,context,session,fields,ctx_names=m5.load_gc_native_rows(source,base)
        native_decisions=times+300*10**9
        f=pd.read_parquet(inputs['F']['path']);lev=pd.read_parquet(inputs['L']['path'])
        for t in (f,lev):
            if not np.array_equal(t.decision_ns,native_decisions):raise RuntimeError('P0_DIAG_FULL_CLOCK')
        common=f.ofi_valid.to_numpy()&f.bbo_valid.to_numpy()&lev.rolling15_valid.to_numpy()
        prep=binding_read(recipe['inputs']['preparation'])
        common &= np.isfinite(np.column_stack([f[n].to_numpy() for n in prep['layers']['F']['fields']]+[lev[n].to_numpy() for n in prep['layers']['L']['fields']])).all(axis=1)
        if common.sum()!=37181:raise RuntimeError('P0_DIAG_COHORT')
        rows=np.searchsorted(native_decisions,decisions)
        if not np.array_equal(native_decisions[rows],decisions) or not common[rows].all():raise RuntimeError('P0_DIAG_NATIVE_MAPPING')
        rv=snap[:,fields.index(recipe['rule']['V_state_field'])];trend=np.sign(snap[:,fields.index(recipe['rule']['core_direction_field'])]).astype(int)
        cutoff=pd.Timestamp(recipe['folds'][0]['validation_start_utc_inclusive']).value-86400*10**9
        cuts=np.quantile(rv[native_decisions<cutoff],recipe['rule']['vol_quantiles'])
        if not np.array_equal(cuts,np.array(original['representation']['RV_tercile_cuts_initial_train_only'])):raise RuntimeError('P0_DIAG_REGIME_CUTS')
        rv_state=np.searchsorted(cuts,rv,side='left')
        # Original representativeness used side=left; original rule prediction used side=right.
        # Fix the descriptive regime to the recorded representativeness owner, no model changes.
        native_fold=np.full(len(times),'initial_TRAIN',object)
        for fold in recipe['folds']:
            s=pd.Timestamp(fold['validation_start_utc_inclusive']).value;e=pd.Timestamp(fold['validation_end_utc_exclusive']).value
            native_fold[(native_decisions>=s)&(native_decisions<e)]=fold['fold_id']
        frame['session']=[SESSION_NAME_BY_ID[int(v)] for v in session[rows]];frame['RV_tercile']=rv_state[rows];frame['EMA_trend_sign']=trend[rows]
        if not np.array_equal(native_fold[rows],frame.fold):raise RuntimeError('P0_DIAG_FOLD_MAPPING')
        full_ex=m5.execution_for_horizon_v2(native_decisions,tape,horizon_minutes=30,latency_seconds=60)
        full_valid=m5.exact_intraday_mask_v3(native_decisions,tape,full_ex,horizon_minutes=30)['valid']
        population=pd.DataFrame({'fold':native_fold,'session':[SESSION_NAME_BY_ID[int(v)] for v in session],'RV_tercile':rv_state,'EMA_trend_sign':trend,'matched_input':common,'exact_h30':full_valid,'y30':full_ex.y_mid_bps})
        coverage={}
        for group in (['fold'],['fold','session'],['fold','RV_tercile'],['fold','EMA_trend_sign'],['fold','session','RV_tercile']):
            name='/'.join(group);out=[]
            for labels,g in population.groupby(group,observed=True):
                selected=g[g.matched_input];unselected=g[~g.matched_input]
                out.append({'stratum':[v.item() if isinstance(v,np.generic) else v for v in (labels if isinstance(labels,tuple) else (labels,))],'population':len(g),'matched':len(selected),'coverage':len(selected)/len(g),
                    'selected_markout_mean_bps':float(selected.loc[selected.exact_h30,'y30'].mean()) if len(selected) else None,
                    'unselected_markout_mean_bps':float(unselected.loc[unselected.exact_h30,'y30'].mean()) if len(unselected) else None,
                    'selected_markout_std_bps':float(selected.loc[selected.exact_h30,'y30'].std(ddof=0)) if len(selected) else None,
                    'unselected_markout_std_bps':float(unselected.loc[unselected.exact_h30,'y30'].std(ddof=0)) if len(unselected) else None})
            coverage[name]=out
        keys=['tree_'+a for a in m5.P0_ARMS];good=p0_matched_prediction_mask(frame,keys);metric_frame=frame[good].reset_index(drop=True)
        diagnostic=p0_group_forecasts(metric_frame,keys,plan['quantiles'])
        calibration={};decomposition={};actions={};contrasts={};fixed={};matching={};fixed_trades=[];fixed_daily=[]
        central=recipe['central_scenario'];cost=recipe['economics']['round_trip_execution_cost_bps'];rates=recipe['economics']['annual_financing_cost_bps'];yr=recipe['economics']['seconds_per_year']
        net30=m5.net_cash_values_v3(ex,economics=recipe['economics'])
        evaluation_start=pd.Timestamp(recipe['folds'][0]['validation_start_utc_inclusive']).value;evaluation_end=pd.Timestamp(recipe['TEST_seal_utc']).value
        for key in keys:
            p=frame[key].to_numpy();side=m5.p0_actions(p,frame.spread_at_decision_bps.to_numpy(),recipe['economics'],30)
            if not np.array_equal(side,frame['action_'+key]):raise RuntimeError('P0_DIAG_ACTION_RECONSTRUCTION')
            acts={}
            for fold,g in frame.assign(_p=p,_side=side).groupby('fold'):
                pp=g._p.to_numpy();spread=g.spread_at_decision_bps.to_numpy();long_q=pp-spread-cost-rates[0]*1800/yr;short_q=-pp-spread-cost-rates[1]*1800/yr
                acts[fold]={'rows':len(g),'positive_forecasts':int((pp>0).sum()),'negative_forecasts':int((pp<0).sum()),'LONG_without_execution_fees':int((pp>spread+rates[0]*1800/yr).sum()),
                    'SHORT_without_execution_fees':int((-pp>spread+rates[1]*1800/yr).sum()),'LONG_if_zero_cost_direction_only':int((pp>0).sum()),'SHORT_if_zero_cost_direction_only':int((pp<0).sum()),
                    'actual_LONG_signals':int((g._side==1).sum()),'actual_SHORT_signals':int((g._side==-1).sum()),'FLAT_signals':int((g._side==0).sum()),
                    'min_forecast_bps':float(pp.min()),'max_forecast_bps':float(pp.max()),'max_LONG_net_margin_bps':float(long_q.max()),'max_SHORT_net_margin_bps':float(short_q.max()),
                    'median_spread_bps':float(np.median(spread)),'min_required_abs_SHORT_bps':float(np.min(spread+cost+rates[1]*1800/yr))}
            actions[key]=acts
            # Descriptive deciles of saved forecasts; never a new trade filter.
            bins=pd.qcut(metric_frame[key],q=plan['calibration_bins'],duplicates='drop');items=[]
            for label,g in metric_frame.groupby(bins,observed=True):
                items.append({'prediction_bin':str(label),**p0_forecast_diagnostics(g.y30_mid_bps,g[key],g.train_mean_bps,plan['quantiles'])})
            calibration[key]=items
            y=metric_frame.y30_mid_bps.to_numpy();pp=metric_frame[key].to_numpy();c=metric_frame.train_mean_bps.to_numpy()
            cov=float(np.mean((y-y.mean())*(pp-pp.mean())));within=between=0.
            for _,g in metric_frame.groupby('fold'):
                w=len(g)/len(metric_frame);yy=g.y30_mid_bps.to_numpy();pred=g[key].to_numpy()
                within+=w*float(np.mean((yy-yy.mean())*(pred-pred.mean())))
                between+=w*(float(yy.mean())-float(y.mean()))*(float(pred.mean())-float(pp.mean()))
            if not np.isclose(within+between,cov,atol=1e-12,rtol=0):raise RuntimeError('P0_DIAG_COVARIANCE_IDENTITY')
            decomposition[key]={'within_fold_covariance_bps2':within,'between_fold_covariance_bps2':between,'pooled_covariance_bps2':cov}
            if key!='tree_CORE':
                core=metric_frame.tree_CORE.to_numpy();increment=pp-core;residual=y-core;loss=(y-core)**2-(y-pp)**2
                records=[{'entry_ns':int(t),'v':float(v)} for t,v in zip(metric_frame.decision_ns,loss)]
                contrasts[key]={'prediction_increment_std_bps':float(increment.std()),'mean_residual_alignment_bps2':float(np.mean(residual*increment)),
                    'increment_energy_bps2':float(np.mean(increment**2)),'MSE_improvement_bps2':float(loss.mean()),
                    'paired_weekly_interval':m5.weekly_block_bootstrap(records,'v',reps=recipe['bootstrap_reps'],seed=recipe['bootstrap_seed'],levels=recipe['confidence_levels']),
                    'fold_MSE_improvement_bps2':{str(f):float(np.mean((g.y30_mid_bps-g.tree_CORE)**2-(g.y30_mid_bps-g[key])**2)) for f,g in metric_frame.groupby('fold')}}
            tr=old_trades[old_trades.strategy==key].copy();at=np.searchsorted(ex.entry_ns,tr.entry_ns)
            if not np.array_equal(ex.entry_ns[at],tr.entry_ns):raise RuntimeError('P0_DIAG_ENTRY_MATCH')
            selected_net=net30[at,np.where(tr.side.to_numpy()>0,0,1)]
            rec=[{'entry_ns':int(t),'v':float(v)} for t,v in zip(tr.entry_ns,selected_net)]
            events=sorted([(int(t),1) for t in tr.entry_ns]+[(int(t),-1) for t in ex.exit_ns[at]])
            balance=peak=0
            for _,v in events:balance+=v;peak=max(peak,balance)
            matching[key]={'entries':len(tr),'dynamic_net_bps':original['cells'][key]['economics'][central]['net_total_bps'],
                'same_entries_h30_net_sum_bps_not_portfolio':float(selected_net.sum()),'same_entries_h30_mean_net_bps':float(selected_net.mean()) if len(tr) else None,
                'same_entries_h30_weekly_mean_CI':m5.weekly_block_bootstrap(rec,'v',reps=recipe['bootstrap_reps'],seed=recipe['bootstrap_seed'],levels=plan['holding_confidence_levels']),
                'overlapping_positions_if_same_entries_all_held30':peak,'dynamic_hold_minutes_quantiles':{str(q):float(v) for q,v in zip(plan['quantiles'],np.quantile(tr.elapsed_seconds/60,plan['quantiles']))},
                'dynamic_gross_sum_bps':float(tr.gross_bps.sum()),'h30_gross_sum_bps':float(np.where(tr.side.to_numpy()>0,ex.long_gross_bps[at],ex.short_gross_bps[at]).sum())}
            order_t,order_s,chosen=p0_fixed_horizon_orders(decisions,side,ex)
            sums,daily,ledger,account=m5.p0_position_ledger(order_t,order_s,tape,evaluation_start,evaluation_end,recipe['cost_scenarios'],yr)
            # Scheduled diagnostic exits must be exactly30min and closed pre-seal.
            if len(ledger)!=len(chosen) or any(t['elapsed_seconds']!=1800 or t['terminal_open'] for t in ledger):raise RuntimeError('P0_DIAG_FIXED_HOLD_LEDGER')
            expected30=float(net30[chosen,np.where(side[chosen]>0,0,1)].sum())
            if not np.isclose(sums[central]['net_total_bps'],expected30,rtol=0,atol=1e-8):raise RuntimeError('P0_FIXED30_CANONICAL_CASH_PARITY')
            fixed[key]={'economics':sums,'ledger':account,'new_entry_count':len(chosen),'per_fold_entries':frame.iloc[chosen].groupby('fold').size().to_dict()}
            dynamic=old_daily[(old_daily.strategy==key)&(old_daily.scenario==central)].set_index('day').equity_bps
            dyn_index=pd.DatetimeIndex(dynamic.index);dynamic.index=dyn_index
            diff=daily[central]-dynamic
            if diff.isna().any():raise RuntimeError('P0_DIAG_EQUITY_TIME_MATCH')
            change=diff.diff();change.iloc[0]=diff.iloc[0]
            rec=[{'entry_ns':int(t.value),'v':float(v)} for t,v in change.items()]
            fixed[key]['paired_dynamic_to_fixed_daily_net_CI']=m5.weekly_block_bootstrap(rec,'v',reps=recipe['bootstrap_reps'],seed=recipe['bootstrap_seed'],levels=plan['holding_confidence_levels'])
            for name,series in daily.items():fixed_daily.append(pd.DataFrame({'strategy':key,'scenario':name,'day':series.index,'equity_bps':series.to_numpy()}))
            fixed_trades.extend([dict(t,strategy=key) for t in ledger])
        frame.to_parquet(stage/'DECISION_DIAGNOSTICS.parquet',index=False)
        pd.DataFrame(fixed_trades).to_parquet(stage/'FIXED30_DIAGNOSTIC_TRADES.parquet',index=False)
        pd.concat(fixed_daily,ignore_index=True).to_parquet(stage/'FIXED30_DIAGNOSTIC_DAILY_EQUITY.parquet',index=False)
        result={'status':'P0_CLOSING_DIAGNOSIS_COMPLETE_NO_REFIT','source_head':binding['source_head'],'plan_sha256':binding['plan_sha256'],
            'development_decisions':len(frame),'primary_tree_common_metric_rows':len(metric_frame),'full_native_rows':len(population),'F_L_input_rows':int(common.sum()),
            'forecast_diagnostics':diagnostic,'calibration_deciles_descriptive_only':calibration,'within_between_fold_covariance':decomposition,
            'action_funnel':actions,'incremental_saved_prediction_value':contrasts,'same_entry_30min_markouts':matching,'fixed30_one_position_diagnostic':fixed,
            'coverage':coverage,'RV_cuts_existing_initial_TRAIN':cuts.tolist(),'models_refitted':0,'new_thresholds_or_horizons_selected':False,
            'raw_h30_label_reconstruction':'EXACT_ALL17301','native_TEST_accessed':False,'historical_cost_truth':False,'source_unchanged':True,'limits':proof,'elapsed_seconds':time.monotonic()-started,
            'inference_limits':plan['inference_limits'],'go_no_go_rule':plan['go_no_go_rule']}
        for item in plan['inputs'].values():binding_read(item)
        all_bindings=[source['signal_manifest'],source['feature_inputs']['parquet'],source['reference']['data']['execution_tape']['manifest']]+source['execution_tape_partitions']
        all_bindings += [b for split in source['native_datasets'].values() for b in split.values()]
        for b in all_bindings:binding_read(b)
        for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_DIAG_CODE_POST')
        io.verify_digest(planpath,binding['plan_sha256'],'P0_DIAG_PLAN_POST');io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_DIAG_POLICY_POST')
        io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_DIAG_APPROVAL_POST');io.verify_digest(binding_path,io.read_json(root/'EXECUTION_CLAIM.json')['binding_sha256'],'P0_DIAG_BINDING_POST')
        if io.git_head()!=binding['source_head']:raise RuntimeError('P0_DIAG_HEAD_POST')
        io.write_json(stage/'RESULT.json',result);inventory={}
        for p in stage.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
            inventory[p.name]={'sha256':io.sha256_file(p),'bytes':p.stat().st_size}
        io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'result_sha256':io.sha256_file(stage/'RESULT.json'),'artifacts':inventory,'source_unchanged':True,'test_data_used':False,'models_refitted':0,'finished_utc':datetime.now(timezone.utc).isoformat()})
        for name,b in inventory.items():io.verify_digest(stage/name,b['sha256'],'P0_DIAG_ARTIFACT')
        fd=os.open(stage,os.O_DIRECTORY|os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        publish_bundle_directory_noreplace(stage,root/'RESULTS');print('P0_CLOSING_DIAGNOSIS_COMPLETE',flush=True)
    except Exception as exc:
        if not (root/'FAILURE.json').exists():io.write_json(root/'FAILURE.json',{'status':'FAILED_NO_ADMISSION','error':str(exc),'exception':type(exc).__name__})
        raise


GC_EVENT_NUMERIC_PLAN_ID = 'NATIVE_M5_GC_CLOCK_NUMERIC_REFERENCE_20261020_118'
GC_EVENT_NUMERIC_ROOT = Path('/home/andre2/GX1_RUNS') / GC_EVENT_NUMERIC_PLAN_ID
GC_EVENT_NUMERIC_POLICY_KEY = 'gc_clock_numeric_reference_118'
GC_EVENT_CLOCK_DOMAINS = ('send_to_capture', 'receive_minus_event', 'matching_clock')


def reference_event_clock_means(minutes, selected, events, timeframe):
    """Independent arithmetic on published M1 float64 sums, not raw ns truth.

    Only duration/session constants are shared with production. Integer flooring
    and math.fsum do not call its label/aggregation/reduceat implementations.
    The error envelope is the float64 forward bound gamma_(k-1)*sum(abs(s_i)),
    plus rounding of fsum and the two quotients; no data-fitted tolerance.
    """
    import math
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    minutes = np.asarray(minutes, np.int64); selected = np.asarray(selected)
    duration = MULTI_TF_SHIFT[timeframe].value
    offset = TRADING_SESSION_BOUNDARY_OFFSET.value % duration
    if (minutes.ndim != 1 or not len(minutes) or selected.shape != minutes.shape
            or np.any(np.diff(minutes) <= 0) or np.any(minutes % MINUTE)
            or np.any(~np.isfinite(selected)) or np.any(selected != np.floor(selected))):
        raise RuntimeError('GC_EVENT_REFERENCE_M1_GRID_OR_CONTRACT')
    required = ['qualified_final_event_count'] + [d + suffix for d in GC_EVENT_CLOCK_DOMAINS
        for suffix in ('_count', '_sum_ns')]
    arrays = {name: np.asarray(events[name], np.float64) for name in required}
    if any(v.shape != minutes.shape or np.isinf(v).any() for v in arrays.values()):
        raise RuntimeError('GC_EVENT_REFERENCE_M1_SHAPE_OR_INFINITY')
    labels = ((minutes - offset) // duration) * duration + offset
    stamps, first, sizes = np.unique(labels, return_index=True, return_counts=True)
    out = {'bar_start_ns': stamps, 'available_ns': stamps + duration,
        'observed_event_valid': np.zeros(len(stamps), bool),
        'observed_event_accepted_minutes': np.zeros(len(stamps), np.int64)}
    for domain in GC_EVENT_CLOCK_DOMAINS:
        prefix = 'observed_event_' + domain
        for suffix in ('_count', '_sum_ns', '_mean_ns', '_sum_bound', '_mean_bound', '_mean_of_means'):
            out[prefix + suffix] = np.full(len(stamps), np.nan)
        out[prefix + '_valid'] = np.zeros(len(stamps), bool)
        out[prefix + '_unequal_positive_counts'] = np.zeros(len(stamps), bool)
    unit = np.finfo(np.float64).eps / 2
    for row, (stamp, start, size) in enumerate(zip(stamps, first, sizes)):
        stop = start + size; sl = slice(start, stop)
        qualified = arrays['qualified_final_event_count'][sl]
        accepted = np.isfinite(qualified) & (qualified > 0)
        out['observed_event_accepted_minutes'][row] = int(accepted.sum())
        contracts = selected[sl]
        valid = (size == duration // MINUTE and minutes[start] == stamp
            and minutes[stop-1] == stamp + duration - MINUTE
            and np.all(contracts >= 0) and np.all(contracts == contracts[0]) and accepted.any())
        out['observed_event_valid'][row] = valid
        if not valid:
            continue
        for domain in GC_EVENT_CLOCK_DOMAINS:
            prefix = 'observed_event_' + domain
            counts = arrays[domain + '_count'][sl][accepted]
            sums = arrays[domain + '_sum_ns'][sl][accepted]
            if (not np.isfinite(counts).all() or not np.isfinite(sums).all()
                    or np.any(counts < 0) or np.any(counts != np.floor(counts))
                    or np.any((counts == 0) & (sums != 0))):
                raise RuntimeError('GC_EVENT_REFERENCE_M1_DOMAIN:' + domain)
            count = sum(int(n) for n in counts)
            if count > 2**53:
                raise RuntimeError('GC_EVENT_REFERENCE_COUNT_FLOAT64_EXACTNESS')
            total = math.fsum(float(v) for v in sums)
            magnitude = math.fsum(abs(float(v)) for v in sums)
            k = len(sums); gamma = (k-1)*unit / (1-(k-1)*unit)
            bound = gamma*magnitude + unit*abs(total)
            out[prefix + '_count'][row] = count
            out[prefix + '_sum_ns'][row] = total
            out[prefix + '_sum_bound'][row] = bound
            out[prefix + '_valid'][row] = count > 0
            if count:
                mean = total/count
                out[prefix + '_mean_ns'][row] = mean
                # Bounds both stored production quotient and reference quotient.
                out[prefix + '_mean_bound'][row] = (bound/count + 2*unit*abs(mean))/(1-unit)
                positive = counts > 0
                out[prefix + '_mean_of_means'][row] = math.fsum(
                    float(s)/int(n) for s, n in zip(sums[positive], counts[positive])) / int(positive.sum())
                out[prefix + '_unequal_positive_counts'][row] = len(set(map(int, counts[positive]))) > 1
    return out


def compare_event_clock_reference(reference, actual):
    """Check masks, counts, float64 forward bounds and discriminating coverage."""
    for name in ('bar_start_ns', 'available_ns', 'observed_event_valid', 'observed_event_accepted_minutes'):
        if not np.array_equal(reference[name], np.asarray(actual[name])):
            raise RuntimeError('GC_EVENT_NUMERIC_GRID_OR_MASK:' + name)
    report = {'buckets': len(reference['bar_start_ns']), 'domains': {}}
    for domain in GC_EVENT_CLOCK_DOMAINS:
        prefix = 'observed_event_' + domain; valid = reference[prefix + '_valid']
        if not np.array_equal(valid, np.asarray(actual[prefix + '_valid'])):
            raise RuntimeError('GC_EVENT_NUMERIC_DOMAIN_VALIDITY:' + domain)
        errors = {}
        for suffix, bound_suffix in (('_count', None), ('_sum_ns', '_sum_bound'), ('_mean_ns', '_mean_bound')):
            expected = reference[prefix + suffix]; observed = np.asarray(actual[prefix + suffix], np.float64)
            if np.isinf(observed).any() or not np.array_equal(np.isnan(expected), np.isnan(observed)):
                raise RuntimeError('GC_EVENT_NUMERIC_FINITE_MASK:' + prefix + suffix)
            finite = np.isfinite(expected); delta = np.abs(observed[finite] - expected[finite])
            bound = 0 if bound_suffix is None else reference[prefix + bound_suffix][finite]
            if np.any(delta > bound):
                raise RuntimeError('GC_EVENT_NUMERIC_VALUE:' + prefix + suffix)
            if finite.any(): errors[suffix.lstrip('_')] = float(delta.max())
        if valid.any():
            quotient = np.asarray(actual[prefix + '_sum_ns'])[valid] / np.asarray(actual[prefix + '_count'])[valid]
            if not np.array_equal(quotient, np.asarray(actual[prefix + '_mean_ns'])[valid]):
                raise RuntimeError('GC_EVENT_NUMERIC_STORED_RATIO:' + domain)
        unequal = valid & reference[prefix + '_unequal_positive_counts']
        discriminating = unequal & (np.abs(reference[prefix + '_mean_of_means'] - reference[prefix + '_mean_ns'])
            > reference[prefix + '_mean_bound'])
        if not discriminating.any():
            raise RuntimeError('GC_EVENT_NUMERIC_NO_UNEQUAL_COUNT_WITNESS:' + domain)
        row = int(np.flatnonzero(discriminating)[0])
        report['domains'][domain] = {'observed_buckets': int(valid.sum()),
            'unequal_positive_count_buckets': int(unequal.sum()),
            'discriminating_buckets': int(discriminating.sum()), 'max_absolute_error': errors,
            'first_witness': {'bar_start_ns': int(reference['bar_start_ns'][row]),
                'count': float(reference[prefix + '_count'][row]),
                'sum_ns': float(reference[prefix + '_sum_ns'][row]),
                'reference_mean_ns': float(reference[prefix + '_mean_ns'][row]),
                'mean_of_means_ns': float(reference[prefix + '_mean_of_means'][row]),
                'mean_error_bound_ns': float(reference[prefix + '_mean_bound'][row])}}
    return report


GC_EVENT_BOOK_PLAN_ID = 'NATIVE_M5_GC_EVENT_BOOK_NUMERIC_REFERENCE_20261020_121'
GC_EVENT_BOOK_POLICY_KEY = 'gc_event_book_numeric_reference_121'
# Independent declarations; tests compare these to the actual executable owners.
GC_EVENT_REFERENCE_REDUCERS = {
    'qualified_final_event_count': ('sum','event_count'),
    'spread_state_min_ticks': ('min','ticks'), 'spread_state_max_ticks': ('max','ticks'),
    **{name:('sum','event_count') for name in ('spread_change_count','spread_expansion_count',
        'spread_contraction_count','ts_in_delta_clamped_low_count','ts_in_delta_clamped_high_count',
        'undefined_event_clock_count','negative_send_to_capture_count','negative_matching_clock_count')},
    **{name:('sum','ticks') for name in ('spread_expansion_sum_ticks','spread_contraction_sum_ticks')},
    **{name:('max','ticks') for name in ('spread_max_expansion_ticks','spread_max_contraction_ticks')},
    **{domain+suffix:(operation,unit) for domain in ('receive_interval',)+GC_EVENT_CLOCK_DOMAINS
        for suffix,operation,unit in (('_count','sum','event_count'),('_sum_ns','sum','nanoseconds'),
            ('_min_ns','min','nanoseconds'),('_max_ns','max','nanoseconds'))}}
GC_BOOK_REFERENCE_ADDITIVE = ('ofi_contracts','bid_refill_contracts','ask_refill_contracts',
    'bid_depletion_contracts','ask_depletion_contracts','quote_updates')
GC_BOOK_REFERENCE_MEANS = {'tw_imbalance':'imbalance','tw_spread_ticks':'ticks',
    'tw_bid_contracts':'displayed_contracts','tw_ask_contracts':'displayed_contracts',
    'tw_microprice_offset_ticks':'ticks'}
GC_ORDER_REFERENCE_MEANS = {'tw_bid_orders':'displayed_orders','tw_ask_orders':'displayed_orders',
    'tw_order_count_imbalance':'imbalance','tw_bid_contracts_per_order':'contracts_per_order',
    'tw_ask_contracts_per_order':'contracts_per_order'}
GC_EVENT_BOOK_AUXILIARY_COLUMNS = frozenset([
    'observed_event_'+d+'_sum_ns' for d in ('receive_interval',)+GC_EVENT_CLOCK_DOMAINS] + [
    'observed_event_qualified_covered_ns','observed_book_clock_minutes','observed_book_expected_clock_minutes',
    'observed_book_covered_ns','observed_order_covered_ns','observed_book_first_minute_ns','observed_book_last_minute_ns'])


def event_book_reference_units():
    units = {'bar_start_ns':'utc_nanoseconds_metadata','available_ns':'utc_nanoseconds_metadata',
        'contract_index':'contract_index_metadata','observed_event_valid':'binary',
        'observed_event_accepted_minutes':'minutes','observed_event_qualified_covered_ns':'nanoseconds',
        'observed_event_qualified_events_per_observed_second':'events_per_observed_second',
        'observed_event_coverage_fraction':'fraction'}
    units.update({'observed_event_'+name:unit for name,(_,unit) in GC_EVENT_REFERENCE_REDUCERS.items()})
    for d in ('receive_interval',)+GC_EVENT_CLOCK_DOMAINS:
        units.update({'observed_event_'+d+'_mean_ns':'nanoseconds','observed_event_'+d+'_valid':'binary'})
    units.update({'observed_book_'+name:'binary' for name in ('contract_valid','flow_valid','valid','response_contiguous')})
    units['observed_order_valid']='binary'
    for prefix in ('observed_book_flow_','observed_book_','observed_order_'):
        units[prefix+'accepted_minutes']='minutes'; units[prefix+'coverage_fraction']='fraction'
    for name in ('clock_minutes','expected_clock_minutes','response_span_minutes'):
        units['observed_book_'+name]='minutes'
    for prefix in ('observed_book_','observed_order_'):units[prefix+'covered_ns']='nanoseconds'
    units.update({'observed_book_'+name:('state_change_count' if name=='quote_updates' else 'contracts')
        for name in GC_BOOK_REFERENCE_ADDITIVE})
    units.update({'observed_book_'+name:unit for name,unit in GC_BOOK_REFERENCE_MEANS.items()})
    units.update({'observed_order_'+name:unit for name,unit in GC_ORDER_REFERENCE_MEANS.items()})
    for name in ('opening_mid_tick','closing_mid_tick','mid_response_ticks'):units['observed_book_'+name]='ticks'
    for name in ('first_minute_ns','last_minute_ns'):units['observed_book_'+name]='utc_nanoseconds_metadata'
    return units


def event_book_numeric_active_fields():
    columns = set(event_book_reference_units()) - GC_EVENT_BOOK_AUXILIARY_COLUMNS - {
        'bar_start_ns','available_ns','contract_index'}
    fields = {tf:{column:'gc_'+tf.lower()+'_'+column for column in sorted(columns)}
        for tf in ('M5','M15','H1','H4','D1')}
    strict = ('ofi_valid','bbo_valid','order_count_valid') + GC_BOOK_REFERENCE_ADDITIVE + tuple(
        GC_BOOK_REFERENCE_MEANS) + ('mid_response_ticks',) + tuple(GC_ORDER_REFERENCE_MEANS)
    fields['strict_m5'] = {column:'gc_book_'+column for column in strict}
    return fields


def _event_book_sum(values):
    import math
    values = list(map(float,values)); total = math.fsum(values)
    unit = np.finfo(np.float64).eps/2; k = len(values)
    gamma = max(0,k-1)*unit/(1-max(0,k-1)*unit)
    return total, gamma*math.fsum(map(abs,values)) + unit*abs(total)


def _event_book_ratio(numerator, denominator, numerator_bound=0., denominator_bound=0.):
    if denominator <= 0:return np.nan, 0.
    if denominator <= denominator_bound:raise RuntimeError('GC_EVENT_BOOK_UNCERTAIN_DENOMINATOR')
    value = numerator/denominator; unit=np.finfo(np.float64).eps/2
    return value, (numerator_bound+abs(value)*denominator_bound)/(denominator-denominator_bound) + 3*unit*abs(value)/(1-3*unit)


def reference_event_book_aggregates(minutes, selected, events, book, timeframe):
    """Own M1-to-TF arithmetic, preserving event/flow/BBO/order/strict domains."""
    from gx1.features.htf_features import MULTI_TF_SHIFT
    clock=reference_event_clock_means(minutes,selected,events,timeframe)
    minutes=np.asarray(minutes,np.int64);selected=np.asarray(selected,np.int64)
    duration=MULTI_TF_SHIFT[timeframe].value; expected=duration//MINUTE
    stamps=clock['bar_start_ns']
    starts=np.searchsorted(minutes,stamps); stops=np.searchsorted(minutes,stamps+duration)
    units=event_book_reference_units()
    out={name:np.full(len(stamps),False if unit=='binary' else np.nan,
        dtype=bool if unit=='binary' else np.float64) for name,unit in units.items()}
    out['bar_start_ns']=stamps;out['available_ns']=stamps+duration
    bounds={name:np.zeros(len(stamps)) for name in out}
    required=('ofi_valid','bbo_valid','order_count_valid','covered_ns','order_covered_ns',
        'opening_mid_tick','closing_mid_tick')+GC_BOOK_REFERENCE_ADDITIVE+tuple(GC_BOOK_REFERENCE_MEANS)+tuple(GC_ORDER_REFERENCE_MEANS)
    arrays={name:np.asarray(book[name]) for name in required}
    if any(v.shape!=minutes.shape for v in arrays.values()):raise RuntimeError('GC_EVENT_BOOK_M1_SHAPE')
    for name in ('ofi_valid','bbo_valid','order_count_valid'):
        if not np.isin(arrays[name],(0,1)).all():raise RuntimeError('GC_EVENT_BOOK_M1_MASK')
    flow=arrays['ofi_valid'].astype(bool); bbo=arrays['bbo_valid'].astype(bool); orders=arrays['order_count_valid'].astype(bool)
    covered=arrays['covered_ns']; order_covered=arrays['order_covered_ns']
    if (not np.isfinite(covered).all() or not np.isfinite(order_covered).all()
            or np.any(covered<0) or np.any(covered>MINUTE) or np.any(order_covered<0)
            or np.any(order_covered>covered) or np.any(covered!=np.floor(covered))
            or np.any(order_covered!=np.floor(order_covered)) or np.any(bbo & (covered!=MINUTE))
            or np.any(orders & (order_covered!=MINUTE)) or np.any(bbo & ~flow) or np.any(orders & ~bbo)):
        raise RuntimeError('GC_EVENT_BOOK_M1_COVERAGE_OR_MASK_DEPENDENCY')
    for names,mask in ((GC_BOOK_REFERENCE_ADDITIVE,flow),
            (tuple(GC_BOOK_REFERENCE_MEANS)+('opening_mid_tick','closing_mid_tick'),bbo),
            (tuple(GC_ORDER_REFERENCE_MEANS),orders)):
        if any(not np.isfinite(arrays[name][mask]).all() for name in names):
            raise RuntimeError('GC_EVENT_BOOK_M1_QUALIFIED_NONFINITE')
    strict_units={name:('binary' if name.endswith('_valid') else 'ticks' if name=='mid_response_ticks'
        else units.get('observed_book_'+name,units.get('observed_order_'+name)))
        for name in event_book_numeric_active_fields()['strict_m5']}
    strict={name:np.full(len(stamps),False if unit=='binary' else np.nan,
        dtype=bool if unit=='binary' else np.float64) for name,unit in strict_units.items()} if timeframe=='M5' else None
    strict_bounds={name:np.zeros(len(stamps)) for name in strict} if strict is not None else None
    for row,(stamp,start,stop) in enumerate(zip(stamps,starts,stops)):
        sl=slice(start,stop); size=stop-start; contracts=selected[sl]
        same=bool(size and np.all(contracts>=0) and np.all(contracts==contracts[0]))
        full=size==expected and minutes[start]==stamp and minutes[stop-1]==stamp+duration-MINUTE
        bucket=bool(full and same); ev=bool(clock['observed_event_valid'][row])
        out['contract_index'][row]=contracts[0]
        out['observed_event_valid'][row]=ev
        out['observed_event_accepted_minutes'][row]=clock['observed_event_accepted_minutes'][row]
        accepted=np.isfinite(events['qualified_final_event_count'][sl]) & (events['qualified_final_event_count'][sl]>0)
        if ev:
            for name,(operation,_) in GC_EVENT_REFERENCE_REDUCERS.items():
                values=np.asarray(events[name][sl]); values=values[accepted & np.isfinite(values)]
                key='observed_event_'+name
                if operation=='sum':out[key][row],bounds[key][row]=_event_book_sum(values)
                elif len(values):out[key][row]=float(min(values) if operation=='min' else max(values))
            values=np.asarray(events['qualified_covered_ns'][sl])[accepted]
            if np.isfinite(values).all():
                total,error=_event_book_sum(values);out['observed_event_qualified_covered_ns'][row]=total
                bounds['observed_event_qualified_covered_ns'][row]=error
                key='observed_event_coverage_fraction';out[key][row],bounds[key][row]=_event_book_ratio(total,duration,error)
                key='observed_event_qualified_events_per_observed_second'
                nkey='observed_event_qualified_final_event_count'
                n=out[nkey][row]*NS;nb=bounds[nkey][row]*NS+np.finfo(np.float64).eps/2*abs(n)
                out[key][row],bounds[key][row]=_event_book_ratio(n,total,nb,error)
            for domain in ('receive_interval',)+GC_EVENT_CLOCK_DOMAINS:
                prefix='observed_event_'+domain; count=out[prefix+'_count'][row]
                out[prefix+'_valid'][row]=count>0
                out[prefix+'_mean_ns'][row],bounds[prefix+'_mean_ns'][row]=_event_book_ratio(
                    out[prefix+'_sum_ns'][row],count,bounds[prefix+'_sum_ns'][row],bounds[prefix+'_count'][row])
        f=flow[sl];b=bbo[sl];o=orders[sl]
        bt=sum(int(v) for v in covered[sl][b]);ot=sum(int(v) for v in order_covered[sl][o])
        fv=bucket and bool(f.any());bv=bucket and bt>0;ov=bucket and ot>0
        for name,value in {'contract_valid':same,'clock_minutes':size,'expected_clock_minutes':expected,
                'flow_valid':fv,'flow_accepted_minutes':int(f.sum()),'flow_coverage_fraction':float(f.sum())/expected,
                'valid':bv,'accepted_minutes':int(b.sum()),'covered_ns':bt,'coverage_fraction':bt/duration}.items():
            out['observed_book_'+name][row]=value
        for name,value in {'valid':ov,'accepted_minutes':int(o.sum()),'covered_ns':ot,'coverage_fraction':ot/duration}.items():
            out['observed_order_'+name][row]=value
        for name in GC_BOOK_REFERENCE_ADDITIVE:
            if fv:out['observed_book_'+name][row],bounds['observed_book_'+name][row]=_event_book_sum(arrays[name][sl][f])
        for names,mask,weights,total,valid,prefix in ((GC_BOOK_REFERENCE_MEANS,b,covered[sl],bt,bv,'observed_book_'),
                (GC_ORDER_REFERENCE_MEANS,o,order_covered[sl],ot,ov,'observed_order_')):
            if valid:
                for name in names:
                    weighted,error=_event_book_sum(float(v)*int(w) for v,w in zip(arrays[name][sl][mask],weights[mask]))
                    out[prefix+name][row],bounds[prefix+name][row]=_event_book_ratio(weighted,total,error)
        if bv:
            indices=np.flatnonzero(b)+start; first,last=int(indices[0]),int(indices[-1])
            opening=float(arrays['opening_mid_tick'][first]);closing=float(arrays['closing_mid_tick'][last])
            span=(int(minutes[last])-int(minutes[first]))//MINUTE+1
            for name,value in {'opening_mid_tick':opening,'closing_mid_tick':closing,'mid_response_ticks':closing-opening,
                    'response_span_minutes':span,'response_contiguous':int(b.sum())==span,
                    'first_minute_ns':float(minutes[first]),'last_minute_ns':float(minutes[last])}.items():
                out['observed_book_'+name][row]=value
        if strict is not None:
            sf=bucket and bool(f.all());sb=bucket and bool(b.all());so=bucket and bool(o.all())
            for name,value in (('ofi_valid',sf),('bbo_valid',sb),('order_count_valid',so)):strict[name][row]=value
            for name in GC_BOOK_REFERENCE_ADDITIVE:
                if sf:strict[name][row],strict_bounds[name][row]=_event_book_sum(arrays[name][sl])
            for names,valid in ((GC_BOOK_REFERENCE_MEANS,sb),(GC_ORDER_REFERENCE_MEANS,so)):
                if valid:
                    for name in names:
                        total,error=_event_book_sum(arrays[name][sl]);strict[name][row],strict_bounds[name][row]=_event_book_ratio(total,expected,error)
            if sb:strict['mid_response_ticks'][row]=float(arrays['closing_mid_tick'][stop-1])-float(arrays['opening_mid_tick'][start])
    return out,bounds,units,clock,(strict,strict_bounds,strict_units)


def compare_event_book_reference(reference, actual, bounds, units):
    if set(reference)!=set(actual):raise RuntimeError('GC_EVENT_BOOK_NUMERIC_COLUMNS')
    report={}
    for name,expected in reference.items():
        observed=np.asarray(actual[name]);expected=np.asarray(expected)
        if (observed.shape!=expected.shape or np.isinf(observed).any()
                or not np.array_equal(np.isnan(expected),np.isnan(observed))):
            raise RuntimeError('GC_EVENT_BOOK_NUMERIC_MASK:'+name)
        finite=np.isfinite(expected);error=np.abs(observed[finite].astype(float)-expected[finite].astype(float))
        limit=bounds[name][finite]
        if not np.isfinite(limit).all() or np.any(limit<0) or np.any(error>limit):
            raise RuntimeError('GC_EVENT_BOOK_NUMERIC_VALUE:'+name)
        item={'unit':units[name],'finite_published_values_compared':int(finite.sum()),
            'nan_masked_values_compared':int((~finite).sum()),'numeric_reference_qualified':bool(finite.any())}
        if finite.any():
            low,high=float(expected[finite].min()),float(expected[finite].max())
            item.update(minimum_reference=low,maximum_reference=high,nonconstant_reference=low!=high,
                maximum_absolute_error=float(error.max()),maximum_forward_roundoff_bound=float(limit.max()))
        report[name]=item
    return report


GC_TRADE_PLAN_ID = 'NATIVE_M5_GC_TRADE_CUMULATIVE_NUMERIC_REFERENCE_20261020_122'
GC_TRADE_POLICY_KEY = 'gc_trade_cumulative_numeric_reference_122'
GC_TRADE_QUANTITIES = ('volume','buy','sell','unknown','trades')
GC_TRADE_SIZE_STATISTICS = ('buy_trade_records','sell_trade_records','unknown_trade_records',
    'size_sum_contracts','size_sum_squared_contracts2','size_min_contracts','size_max_contracts')
GC_TRADE_CVD_FIELDS = ('cvd_segment_delta_contracts','cvd_segment_volume_contracts','cvd_segment_minutes')
GC_TRADE_KNOWN_FIELDS = ('buy_contracts','sell_contracts','unknown_contracts','volume_contracts',
    'known_delta_contracts','unknown_fraction','observed_minutes')
GC_TRADE_AUXILIARY_COLUMNS = ('observed_clock_minutes','observed_expected_clock_minutes',
    'observed_first_minute_ns','observed_last_minute_ns')


def trade_reference_units():
    units={'bar_start_ns':'utc_nanoseconds_metadata','available_ns':'utc_nanoseconds_metadata',
        'contract_index':'contract_index_metadata','volume_valid':'binary','signed_valid':'binary',
        'delta_contracts':'contracts','known_aggressor_fraction':'fraction',
        'observed_valid':'binary','observed_contract_valid':'binary','observed_signed_complete':'binary',
        'observed_avg_executed_trade_contracts':'contracts_per_T_record','observed_known_delta_contracts':'contracts',
        'observed_known_aggressor_fraction':'fraction','observed_accepted_minutes':'minutes',
        'observed_signed_minutes':'minutes','observed_coverage_fraction':'fraction','observed_signed_minute_fraction':'fraction',
        'observed_clock_minutes':'minutes','observed_expected_clock_minutes':'minutes',
        'observed_first_minute_ns':'utc_nanoseconds_metadata','observed_last_minute_ns':'utc_nanoseconds_metadata'}
    for name in GC_TRADE_QUANTITIES:
        unit='T_record_count' if name=='trades' else 'contracts';units[name]=unit
        target={'buy':'buy_known','sell':'sell_known'}.get(name,name);units['observed_'+target]=unit
    units.update({name:'minutes' if name.endswith('_minutes') else 'contracts' for name in GC_TRADE_CVD_FIELDS})
    units.update({'observed_known_cumulative_'+name:('fraction' if name=='unknown_fraction'
        else 'minutes' if name=='observed_minutes' else 'contracts') for name in GC_TRADE_KNOWN_FIELDS})
    for side in ('buy','sell','unknown'):
        units['observed_'+side+'_trade_records']='T_record_count'
        units['observed_'+side+'_avg_executed_trade_contracts']='contracts_per_T_record'
        units['observed_'+side+'_trade_size_valid']='binary'
    for name in ('std','min','max'):units['observed_executed_trade_size_'+name+'_contracts']='contracts'
    return units


def trade_numeric_active_fields():
    observed={name for name in trade_reference_units() if name.startswith('observed_')} - set(GC_TRADE_AUXILIARY_COLUMNS)
    strict=('volume_valid','signed_valid')+GC_TRADE_QUANTITIES+('delta_contracts','known_aggressor_fraction')+GC_TRADE_CVD_FIELDS
    fields={}
    for tf in ('M5','M15','H1','H4','D1'):
        kept=strict if tf in ('M5','M15','H1') else ('volume_valid','volume','unknown','trades','known_aggressor_fraction') if tf=='H4' else ()
        fields[tf]={name:'gc_'+tf.lower()+'_'+name for name in sorted(observed | set(kept))}
    return fields


def reference_trade_cumulatives(minutes, selected, trade, stats):
    """One exact M1 pass; known and fully signed ledgers have distinct resets."""
    minutes=np.asarray(minutes,np.int64);selected=np.asarray(selected)
    if (not len(minutes) or selected.shape!=minutes.shape or np.any(np.diff(minutes)<=0)
            or np.any(minutes%MINUTE) or np.any(selected!=np.floor(selected))):
        raise RuntimeError('GC_TRADE_REFERENCE_M1_GRID')
    if set(stats)!=set(GC_TRADE_SIZE_STATISTICS):raise RuntimeError('GC_TRADE_REFERENCE_STATS_SCHEMA')
    for values in [trade[name] for name in GC_TRADE_QUANTITIES]+list(stats.values()):
        if np.asarray(values).shape!=minutes.shape or np.asarray(values).dtype!=np.dtype('uint64'):
            raise RuntimeError('GC_TRADE_REFERENCE_EXACT_UINT64')
    for name in ('volume_valid','signed_valid'):
        if np.asarray(trade[name]).shape!=minutes.shape or not np.isin(trade[name],(0,1)).all():
            raise RuntimeError('GC_TRADE_REFERENCE_M1_MASK')
    known={name:np.full(len(minutes),np.nan) for name in GC_TRADE_KNOWN_FIELDS}
    cvd={name:np.full(len(minutes),np.nan) for name in GC_TRADE_CVD_FIELDS}
    accum=[0]*5;signed=[0]*3;previous_known=previous_signed=None
    for i,minute in enumerate(minutes):
        volume,buy,sell,unknown,records=(int(trade[name][i]) for name in GC_TRADE_QUANTITIES)
        counts=[int(stats[side+'_trade_records'][i]) for side in ('buy','sell','unknown')]
        if (sum(counts)!=records or int(stats['size_sum_contracts'][i])!=volume
                or volume!=buy+sell+unknown):
            raise RuntimeError('GC_TRADE_REFERENCE_ALL_M1_RECORD_OR_QUANTITY_PARITY')
        if any(count==0 and quantity>0 for count,quantity in zip(counts,(buy,sell,unknown))):
            raise RuntimeError('GC_TRADE_REFERENCE_M1_SIDE_VOLUME_WITHOUT_RECORDS')
        valid=bool(trade['volume_valid'][i]);signed_valid=bool(trade['signed_valid'][i])
        if signed_valid and (not valid or unknown):raise RuntimeError('GC_TRADE_REFERENCE_SIGNED_DOMAIN')
        if valid:
            if selected[i]<0 or volume<=0 or records<=0:raise RuntimeError('GC_TRADE_REFERENCE_ACCEPTED_DOMAIN')
            if (previous_known is None or i!=previous_known+1 or minute-minutes[previous_known]!=MINUTE
                    or selected[i]!=selected[previous_known] or minute//DAY!=minutes[previous_known]//DAY):accum=[0]*5
            accum=[a+b for a,b in zip(accum,(buy,sell,unknown,volume,1))]
            if accum[3]>2**53:raise RuntimeError('GC_TRADE_REFERENCE_FLOAT_QUANTITY_BOUND')
            values=accum[:4]+[accum[0]-accum[1],accum[2]/accum[3],accum[4]]
            for name,value in zip(GC_TRADE_KNOWN_FIELDS,values):known[name][i]=value
            previous_known=i
        if signed_valid:
            if (previous_signed is None or i!=previous_signed+1 or minute-minutes[previous_signed]!=MINUTE
                    or selected[i]!=selected[previous_signed] or minute//DAY!=minutes[previous_signed]//DAY):signed=[0]*3
            signed=[signed[0]+buy-sell,signed[1]+volume,signed[2]+1]
            for name,value in zip(GC_TRADE_CVD_FIELDS,signed):cvd[name][i]=value
            previous_signed=i
    return known,cvd


def reference_trade_aggregates(minutes, selected, trade, stats, known, cvd, timeframe):
    """Independent Python-int totals and variance; no production reducers."""
    import math
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    duration=MULTI_TF_SHIFT[timeframe].value;offset=TRADING_SESSION_BOUNDARY_OFFSET.value%duration
    minutes=np.asarray(minutes,np.int64);selected=np.asarray(selected)
    labels=((minutes-offset)//duration)*duration+offset
    stamps=np.unique(labels);starts=np.searchsorted(minutes,stamps);stops=np.searchsorted(minutes,stamps+duration)
    units=trade_reference_units();expected=duration//MINUTE
    out={name:np.full(len(stamps),False if unit=='binary' else np.nan,
        dtype=bool if unit=='binary' else np.float64) for name,unit in units.items()}
    out['bar_start_ns']=stamps;out['available_ns']=stamps+duration
    bounds={name:np.zeros(len(stamps)) for name in out}
    for row,(stamp,start,stop) in enumerate(zip(stamps,starts,stops)):
        sl=slice(start,stop);size=stop-start;contracts=selected[sl]
        accepted=np.asarray(trade['volume_valid'][sl],bool);signed=np.asarray(trade['signed_valid'][sl],bool)
        same=bool(np.all(contracts>=0) and np.all(contracts==contracts[0]))
        full=size==expected and minutes[start]==stamp and minutes[stop-1]==stamp+duration-MINUTE
        strict=bool(full and same and accepted.all());strict_signed=bool(strict and signed.all())
        number=int(accepted.sum());signed_number=int((accepted & signed).sum());observed=bool(full and same and number)
        for name,value in {'contract_index':contracts[0],'volume_valid':strict,'signed_valid':strict_signed,
                'observed_valid':observed,'observed_contract_valid':same,'observed_clock_minutes':size,
                'observed_expected_clock_minutes':expected,'observed_accepted_minutes':number,
                'observed_signed_minutes':signed_number,'observed_coverage_fraction':number/expected}.items():out[name][row]=value
        if number:out['observed_signed_minute_fraction'][row],bounds['observed_signed_minute_fraction'][row]=_event_book_ratio(signed_number,number)
        totals={name:sum(int(v) for v in trade[name][sl][accepted]) for name in GC_TRADE_QUANTITIES}
        pooled={}
        for name in GC_TRADE_SIZE_STATISTICS:
            values=[int(v) for v in stats[name][sl][accepted]]
            pooled[name]=(min(values) if values else 0) if name=='size_min_contracts' else (max(values) if values else 0) if name=='size_max_contracts' else sum(values)
            if pooled[name]>2**64-1:raise RuntimeError('GC_TRADE_REFERENCE_UINT64_OVERFLOW')
        records=sum(pooled[side+'_trade_records'] for side in ('buy','sell','unknown'))
        if records>2**64-1:raise RuntimeError('GC_TRADE_REFERENCE_UINT64_OVERFLOW')
        variance=records*pooled['size_sum_squared_contracts2']-pooled['size_sum_contracts']**2
        if variance<0:raise RuntimeError('GC_TRADE_REFERENCE_NEGATIVE_INTEGER_VARIANCE')
        if any(value>2**53 for value in totals.values()):raise RuntimeError('GC_TRADE_REFERENCE_FLOAT_QUANTITY_BOUND')
        for name in GC_TRADE_QUANTITIES:
            if strict and (name not in ('buy','sell') or strict_signed):out[name][row]=totals[name]
            if observed:out['observed_'+{'buy':'buy_known','sell':'sell_known'}.get(name,name)][row]=totals[name]
        if strict:
            out['known_aggressor_fraction'][row],bounds['known_aggressor_fraction'][row]=_event_book_ratio(totals['buy']+totals['sell'],totals['volume'])
        if strict_signed:
            out['delta_contracts'][row]=totals['buy']-totals['sell']
            for name in GC_TRADE_CVD_FIELDS:out[name][row]=cvd[name][stop-1]
        for side in ('buy','sell','unknown'):
            count=pooled[side+'_trade_records'];ready=observed and count>0
            if count>2**53:raise RuntimeError('GC_TRADE_REFERENCE_FLOAT_COUNT_BOUND')
            if not count and totals[side]:raise RuntimeError('GC_TRADE_REFERENCE_SIDE_VOLUME_WITHOUT_RECORDS')
            out['observed_'+side+'_trade_size_valid'][row]=ready
            if observed:out['observed_'+side+'_trade_records'][row]=count
            if ready:
                key='observed_'+side+'_avg_executed_trade_contracts'
                out[key][row],bounds[key][row]=_event_book_ratio(totals[side],count)
        if observed:
            positions=np.flatnonzero(accepted)+start;first,last=int(positions[0]),int(positions[-1])
            out['observed_first_minute_ns'][row]=float(minutes[first]);out['observed_last_minute_ns'][row]=float(minutes[last])
            out['observed_signed_complete'][row]=totals['unknown']==0
            out['observed_known_delta_contracts'][row]=totals['buy']-totals['sell']
            for name,numerator,denominator in (
                    ('observed_avg_executed_trade_contracts',totals['volume'],totals['trades']),
                    ('observed_known_aggressor_fraction',totals['buy']+totals['sell'],totals['volume'])):
                out[name][row],bounds[name][row]=_event_book_ratio(numerator,denominator)
            for name in GC_TRADE_KNOWN_FIELDS:
                key='observed_known_cumulative_'+name;out[key][row]=known[name][last]
                if name=='unknown_fraction':bounds[key][row]=np.finfo(np.float64).eps/2*abs(out[key][row])
            value=math.sqrt(float(variance))/records if records else np.nan
            key='observed_executed_trade_size_std_contracts';out[key][row]=value
            # Integer-to-binary64, sqrt and division in BOTH paths: gamma_6.
            unit=np.finfo(np.float64).eps/2;bounds[key][row]=6*unit*abs(value)/(1-6*unit)
            for name in ('min','max'):out['observed_executed_trade_size_'+name+'_contracts'][row]=pooled['size_'+name+'_contracts']
    return out,bounds,units


def compare_trade_reference(reference, actual, bounds, units):
    for name in ('bar_start_ns','available_ns'):
        if (np.asarray(actual[name]).dtype!=np.dtype('int64')
                or not np.array_equal(reference[name],actual[name])):
            raise RuntimeError('GC_TRADE_REFERENCE_EXACT_TF_GRID:'+name)
    contract=np.asarray(actual['contract_index'])
    if (not np.issubdtype(contract.dtype,np.integer)
            or not np.array_equal(np.asarray(reference['contract_index'],np.int64),contract.astype(np.int64))):
        raise RuntimeError('GC_TRADE_REFERENCE_EXACT_TF_GRID:contract_index')
    return compare_event_book_reference(reference,actual,bounds,units)


GC_SMC_PLAN_ID = 'NATIVE_M5_GC_SMC_PRICE_NUMERIC_REFERENCE_20261020_123'
GC_SMC_POLICY_KEY = 'gc_smc_price_numeric_reference_123'
GC_SMC_PRICE_FIELDS = ('open_usd','high_usd','low_usd','close_usd')
GC_SMC_ZONE_FIELDS = ('active','created','first_retouch','low','high','age_bars','origin_volume_contracts')
GC_SMC_FIELDS = ('structure_available','bos_up','bos_down','sweep_up_event','sweep_down_event') + tuple(
    'ob_'+side+'_'+field for side in ('bull','bear') for field in GC_SMC_ZONE_FIELDS) + (
    'ob_bull_input_valid','ob_bear_input_valid')
GC_SMC_M1_COLUMNS = ('bar_start_ns','contract_index','volume_valid','volume',
    'open_tick','high_tick','low_tick','close_tick')
GC_SMC_DIAGNOSTICS = ('boundary_gap_minutes','elapsed_minutes','skipped_candles')


def smc_reference_units():
    units={'bar_start_ns':'utc_nanoseconds_metadata','available_ns':'utc_nanoseconds_metadata',
        'contract_index':'contract_index_metadata','volume_valid':'binary','volume':'contracts',
        'observed_valid':'binary','observed_contract_valid':'binary','observed_volume':'contracts',
        'observed_first_minute_ns':'utc_nanoseconds_metadata','observed_last_minute_ns':'utc_nanoseconds_metadata'}
    for prefix in ('','observed_'):
        units.update({prefix+name:'USD' for name in GC_SMC_PRICE_FIELDS})
    for prefix in ('','observed_proxy_'):
        for name in GC_SMC_FIELDS:
            units[prefix+name]=('USD' if name.endswith(('_low','_high')) else 'contracts' if
                name.endswith('_origin_volume_contracts') else 'observed_bars' if name.endswith('_age_bars') else 'binary')
    units.update({'observed_proxy_'+name:('observed_bars' if name=='skipped_candles' else 'minutes')
        for name in GC_SMC_DIAGNOSTICS})
    units.update({'observed_proxy_ob_'+side+'_state_valid':'binary' for side in ('bull','bear')})
    return units


def smc_numeric_active_fields():
    fields={}
    for tf in ('M5','M15','H1','H4','D1'):
        observed={'observed_proxy_'+name for name in GC_SMC_FIELDS+GC_SMC_DIAGNOSTICS}
        strict=set(GC_SMC_FIELDS) if tf=='M5' else set(GC_SMC_FIELDS)-{'structure_available'} if tf=='M15' else set()
        prices={'observed_'+name for name in GC_SMC_PRICE_FIELDS}
        if tf!='D1':prices.update(GC_SMC_PRICE_FIELDS)
        fields[tf]={name:'gc_'+tf.lower()+'_'+name for name in sorted(observed|strict|prices)}
    return fields,{'close_usd':'gc_book_close_usd'}


def smc_price_reference(ticks, tick_size, price_scale):
    """Rational price oracle; bounds cover both binary64 operation orders.

    TF: integer conversion, scale division, multiplication. Legacy close:
    integer conversion, multiplication, division. Reference rounding adds one
    more operation. gamma_4 is expressed relative to the rounded reference.
    """
    from fractions import Fraction
    value=float(Fraction(int(ticks)*int(tick_size),int(price_scale)))
    unit=np.finfo(np.float64).eps/2
    return value,(4*unit/(1-4*unit))*abs(value)/(1-unit)


def validate_smc_tick_order(maximum_tick, tick_size, price_scale, pivot_tolerance):
    if (not isinstance(tick_size,int) or tick_size<=0 or not isinstance(price_scale,int) or price_scale<=0
            or maximum_tick<0 or maximum_tick>2**53 or not np.isfinite(pivot_tolerance) or pivot_tolerance<=0):
        raise RuntimeError('GC_SMC_REFERENCE_PRICE_DOMAIN')
    maximum,error=smc_price_reference(maximum_tick,tick_size,price_scale)
    step,step_error=smc_price_reference(1,tick_size,price_scale)
    # Owner compares high >= window_max - tolerance (low mirrored). Include
    # subtraction rounding as well as both converted prices; never fit EPS.
    subtraction=np.finfo(np.float64).eps/2*(abs(maximum)+pivot_tolerance)
    if step-step_error-2*error-subtraction<=pivot_tolerance:
        raise RuntimeError('GC_SMC_REFERENCE_TICK_TOLERANCE_NOT_SEPARATED')
    return {'maximum_tick':int(maximum_tick),'minimum_price_step_usd':step,
        'maximum_price_roundoff_bound_usd':error,'pivot_comparison_tolerance_usd':pivot_tolerance,
        'ordering_separation_proven':True}


def reference_smc_sequence(ohlc_ticks, volume, *, zone_reset=None):
    """Independent chronological integer replay; no producer pivot/SMC calls."""
    from gx1.features.smc_v1 import SWING_LOOKBACK
    if SWING_LOOKBACK!=3:raise RuntimeError('GC_SMC_REFERENCE_LOOKBACK_CHANGED')
    candles=np.asarray(ohlc_ticks);quantity=np.asarray(volume);n=len(candles)
    if (candles.shape!=(n,4) or not np.issubdtype(candles.dtype,np.integer) or quantity.shape!=(n,)
            or np.any(quantity<=0) or np.any(quantity>2**53)
            or np.any(candles[:,2]>candles[:,0]) or np.any(candles[:,0]>candles[:,1])
            or np.any(candles[:,2]>candles[:,3]) or np.any(candles[:,3]>candles[:,1])):
        raise RuntimeError('GC_SMC_REFERENCE_CANDLE_DOMAIN')
    reset=np.zeros(n,bool) if zone_reset is None else np.asarray(zone_reset,bool)
    if reset.shape!=(n,):raise RuntimeError('GC_SMC_REFERENCE_RESET_SHAPE')
    out={name:np.full(n,np.nan) for name in GC_SMC_FIELDS}
    pivots={name:np.full(n,-1,np.int64) for name in ('last_high_idx','prev_high_idx','last_low_idx','prev_low_idx')}
    highs=[];lows=[];last={'high':None,'low':None};previous={'high':None,'low':None}
    consumed_bos={'high':None,'low':None};consumed_sweep={'high':None,'low':None}
    origins={'bull':None,'bear':None};zones={'bull':None,'bear':None};known={'bull':True,'bear':True}
    for i,bar in enumerate(candles):
        o,h,l,c=map(int,bar);highs.append(h);lows.append(l)
        center=i-SWING_LOOKBACK
        if center>=SWING_LOOKBACK:
            for side,values,operation in (('high',highs,max),('low',lows,min)):
                # The current prefix ends exactly at the third right candle.
                if values[center]==operation(values[center-SWING_LOOKBACK:i+1]):
                    previous[side],last[side]=last[side],(center,values[center])
        for side in ('high','low'):
            if last[side] is not None:pivots['last_'+side+'_idx'][i]=last[side][0]
            if previous[side] is not None:pivots['prev_'+side+'_idx'][i]=previous[side][0]
        levels=[entry[1] for entry in (*last.values(),*previous.values()) if entry is not None]
        out['structure_available'][i]=float(len(levels)==4 and max(levels)>min(levels))
        events={}
        for side,price,up in (('high',h,True),('low',l,False)):
            level=last[side];identity=None if level is None else level[0]
            bos=level is not None and (c>level[1] if up else c<level[1]) and consumed_bos[side]!=identity
            if bos:consumed_bos[side]=identity
            sweep=level is not None and consumed_bos[side]!=identity and (
                price>level[1] and c<=level[1] if up else price<level[1] and c>=level[1])
            event=sweep and consumed_sweep[side]!=identity
            if event:consumed_sweep[side]=identity
            events[side]=bool(bos)
            out['bos_'+('up' if up else 'down')][i]=float(bos) if level is not None else np.nan
            out['sweep_'+('up' if up else 'down')+'_event'][i]=float(event) if level is not None else np.nan
        if reset[i]:
            origins={'bull':None,'bear':None};zones={'bull':None,'bear':None};known={'bull':False,'bear':False}
        for side,level_side in (('bull','high'),('bear','low')):
            origin=origins[side];created=events[level_side] and origin is not None
            if created:
                zones[side]=[int(candles[origin,2]),int(candles[origin,1]),int(quantity[origin]),i,False]
                known[side]=True
            zone=zones[side];retouch=False
            if zone is not None:
                lower,upper,qty,born,touched=zone
                if c<lower if side=='bull' else c>upper:zones[side]=zone=None
                elif i>born and not touched and l<=upper and h>=lower:retouch=True;zone[4]=True
            ready=last[level_side] is not None;prefix='ob_'+side+'_'
            out[prefix+'input_valid'][i]=float(ready)
            if ready and known[side]:
                for field,value in (('active',zone is not None),('created',created),('first_retouch',retouch)):
                    out[prefix+field][i]=float(value)
                if zone is not None:
                    for field,value in zip(('low','high','origin_volume_contracts','age_bars'),(*zone[:3],i-zone[3])):
                        out[prefix+field][i]=value
        # Current candle can only become an origin for a subsequent candle.
        if c<o:origins['bull']=i
        if c>o:origins['bear']=i
    return out,pivots


def validate_smc_m1_schema(m1):
    if set(m1)!=set(GC_SMC_M1_COLUMNS):raise RuntimeError('GC_SMC_REFERENCE_M1_SCHEMA')
    minutes=np.asarray(m1['bar_start_ns']);shape=minutes.shape
    if (minutes.ndim!=1 or not len(minutes) or minutes.dtype!=np.dtype('int64')
            or any(np.asarray(value).shape!=shape for value in m1.values())
            or not np.issubdtype(np.asarray(m1['contract_index']).dtype,np.integer)
            or np.asarray(m1['volume_valid']).dtype.kind!='b' or np.asarray(m1['volume']).dtype!=np.dtype('uint64')
            or any(not np.issubdtype(np.asarray(m1[name]).dtype,np.integer) for name in GC_SMC_M1_COLUMNS[4:])
            or np.any(np.diff(minutes)<=0)):
        raise RuntimeError('GC_SMC_REFERENCE_M1_SCHEMA')
    admitted=np.asarray(m1['volume_valid']);o,h,l,c=[np.asarray(m1[name])[admitted] for name in GC_SMC_M1_COLUMNS[4:]]
    if (np.any(m1['volume'][admitted]==0) or any(np.any(values<0) or np.any(values>2**53) for values in (o,h,l,c))
            or np.any(l>o) or np.any(o>h) or np.any(l>c) or np.any(c>h)):
        raise RuntimeError('GC_SMC_REFERENCE_ADMITTED_CANDLE_DOMAIN')


def reference_smc_aggregates(m1, timeframe, tick_size, price_scale):
    """M1 integer OHLC/quantity reconstruction followed by independent replay."""
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    duration=MULTI_TF_SHIFT[timeframe].value;offset=TRADING_SESSION_BOUNDARY_OFFSET.value%duration
    validate_smc_m1_schema(m1)
    minutes=np.asarray(m1['bar_start_ns']);selected=np.asarray(m1['contract_index'])
    labels=((minutes-offset)//duration)*duration+offset;stamps=np.unique(labels)
    starts=np.searchsorted(minutes,stamps);stops=np.searchsorted(minutes,stamps+duration);expected=duration//MINUTE
    units=smc_reference_units();out={name:np.full(len(stamps),False if unit=='binary' and name.startswith('observed_proxy_ob_')
        and name.endswith('_state_valid') else np.nan) for name,unit in units.items()}
    for name in ('volume_valid','observed_valid','observed_contract_valid'):out[name]=np.zeros(len(stamps),bool)
    out['bar_start_ns']=stamps;out['available_ns']=stamps+duration;out['contract_index']=np.zeros(len(stamps),np.int64)
    bounds={name:np.zeros(len(stamps)) for name in out}
    ticks={prefix:np.zeros((len(stamps),4),np.int64) for prefix in ('','observed_')}
    for row,(stamp,start,stop) in enumerate(zip(stamps,starts,stops)):
        contracts=selected[start:stop];accepted=np.asarray(m1['volume_valid'][start:stop],bool)
        same=bool(np.all(contracts>=0) and np.all(contracts==contracts[0]))
        full=(stop-start==expected and minutes[start]==stamp and minutes[stop-1]==stamp+duration-MINUTE
            and np.all(np.diff(minutes[start:stop])==MINUTE))
        strict=bool(full and same and accepted.all());observed=bool(full and same and accepted.any())
        out['contract_index'][row]=contracts[0];out['observed_contract_valid'][row]=same
        out['volume_valid'][row]=strict;out['observed_valid'][row]=observed
        for prefix,valid,positions in (('',strict,np.arange(start,stop)),('observed_',observed,np.flatnonzero(accepted)+start)):
            if not valid:continue
            qty=sum(int(v) for v in m1['volume'][positions])
            if qty<=0 or qty>2**53:raise RuntimeError('GC_SMC_REFERENCE_VOLUME_EXACTNESS')
            out[prefix+'volume'][row]=qty
            values=(int(m1['open_tick'][positions[0]]),max(map(int,m1['high_tick'][positions])),
                min(map(int,m1['low_tick'][positions])),int(m1['close_tick'][positions[-1]]))
            ticks[prefix][row]=values
            for name,value in zip(GC_SMC_PRICE_FIELDS,values):
                out[prefix+name][row],bounds[prefix+name][row]=smc_price_reference(value,tick_size,price_scale)
            if prefix:
                for edge,position in (('first',positions[0]),('last',positions[-1])):
                    value=int(minutes[position])
                    if int(float(value))!=value:raise RuntimeError('GC_SMC_REFERENCE_ENDPOINT_PRECISION')
                    out['observed_'+edge+'_minute_ns'][row]=value
    for prefix,valid_name in (('','volume_valid'),('observed_','observed_valid')):
        groups=[];current=[];previous=None;last_segment=None;segment=0
        for row in range(len(stamps)):
            if row and (out['contract_index'][row]!=out['contract_index'][row-1]
                    or not out['observed_contract_valid'][row] or not out['observed_contract_valid'][row-1]):segment+=1
            if not out[valid_name][row]:continue
            continuation=previous is not None and (segment==last_segment if prefix else
                row==previous+1 and stamps[row]-stamps[previous]==duration and segment==last_segment)
            if current and not continuation:groups.append(current);current=[]
            current.append(row);previous=row;last_segment=segment
        if current:groups.append(current)
        for group in groups:
            rows=np.asarray(group,np.int64);reset=np.zeros(len(rows),bool)
            if prefix:
                elapsed=np.diff(stamps[rows]);skipped=elapsed//duration-1;reset[1:]=skipped>0
                for name,values in (('elapsed_minutes',elapsed//MINUTE),('skipped_candles',skipped),
                        ('boundary_gap_minutes',np.maximum((out['observed_first_minute_ns'][rows[1:]]-
                            out['observed_last_minute_ns'][rows[:-1]])/MINUTE-1,0))):
                    out['observed_proxy_'+name][rows[1:]]=values
            replay,_=reference_smc_sequence(ticks[prefix][rows],out[prefix+'volume'][rows],zone_reset=reset)
            target='observed_proxy_' if prefix else ''
            for name,values in replay.items():
                if name.endswith(('_low','_high')):
                    for local in np.flatnonzero(np.isfinite(values)):
                        price,error=smc_price_reference(int(values[local]),tick_size,price_scale)
                        out[target+name][rows[local]]=price;bounds[target+name][rows[local]]=error
                else:out[target+name][rows]=values
            if prefix:
                for side in ('bull','bear'):
                    out['observed_proxy_ob_'+side+'_state_valid'][rows]=np.isfinite(replay['ob_'+side+'_active'])
    return out,bounds,units


def read_smc_numeric_inputs(inputs, start, end, seal):
    values,groups=read_event_numeric_columns(Path(inputs['m1_smc']['path']),list(GC_SMC_M1_COLUMNS),start,end,seal)
    validate_smc_m1_schema(values)
    return values,groups


GC_PROFILE_PLAN_ID = 'NATIVE_M5_GC_PROFILE_FOOTPRINT_NUMERIC_REFERENCE_20261021_124'
GC_PROFILE_POLICY_KEY = 'gc_profile_footprint_numeric_reference_124'
GC_PROFILE_M1_COLUMNS = ('bar_start_ns','contract_index','volume_valid','signed_valid',
    'volume','buy','sell','unknown','close_tick')
GC_PROFILE_LADDER_COLUMNS = ('bar_start_ns','contract_index','price_tick','buy','sell','unknown','signed_valid')
GC_PROFILE_HISTOGRAM_FIELDS = ('vwap_usd','poc_usd','q25_usd','median_usd','q75_usd','price_levels')
GC_PROFILE_FOOTPRINT_FIELDS = ('max_level_delta_contracts','min_level_delta_contracts','buy_poc_usd','sell_poc_usd')
GC_PROFILE_OBSERVED_FOOTPRINT = {'max_level_delta_contracts':'observed_known_max_level_delta_contracts',
    'min_level_delta_contracts':'observed_known_min_level_delta_contracts','buy_poc_usd':'observed_buy_known_poc_usd',
    'sell_poc_usd':'observed_sell_known_poc_usd'}
GC_PROFILE_FAMILIES = ('rolling15','rolling60','anchored_segment','session_observed')
GC_PROFILE_VALUES = ('vwap_usd','poc_usd','val_usd','vah_usd','volume_contracts')


def profile_dynamic_units():
    units={family+'_'+name:('contracts' if name=='volume_contracts' else 'USD')
        for family in GC_PROFILE_FAMILIES for name in GC_PROFILE_VALUES}
    units.update({family+'_valid':'binary' for family in GC_PROFILE_FAMILIES})
    units.update(session_complete_valid='binary',session_accepted_minutes='minutes',session_expected_minutes='minutes')
    units.update({family+'_'+level+'_'+suffix:'ticks' for family in GC_PROFILE_FAMILIES[:3]
        for level in ('vwap','poc','val','vah') for suffix in ('distance_ticks','prior_level_response_ticks')})
    return units


def profile_reference_units():
    units={'bar_start_ns':'utc_nanoseconds_metadata','available_ns':'utc_nanoseconds_metadata',
        'contract_index':'contract_index_metadata'}
    for prefix in ('','observed_'):
        units.update({prefix+name:('price_levels' if name=='price_levels' else 'USD') for name in GC_PROFILE_HISTOGRAM_FIELDS})
    units.update(observed_val_usd='USD',observed_vah_usd='USD')
    for name in GC_PROFILE_FOOTPRINT_FIELDS:
        unit='contracts' if name.endswith('_contracts') else 'USD'
        units[name]=unit;units[GC_PROFILE_OBSERVED_FOOTPRINT[name]]=unit
    units.update({'observed_profile_'+name:unit for name,unit in profile_dynamic_units().items()})
    units.update(observed_profile_source_coverage_fraction='fraction',observed_profile_snapshot_age_minutes='minutes',
        observed_profile_snapshot_available_ns='utc_nanoseconds_metadata')
    return units


def profile_numeric_active_fields():
    fields={}
    observed={name for name in profile_reference_units() if name.startswith('observed_')}
    observed.remove('observed_profile_snapshot_available_ns')
    for tf in ('M5','M15','H1','H4','D1'):
        strict=set(GC_PROFILE_HISTOGRAM_FIELDS) if tf!='D1' else set()
        if tf in ('M5','M15','H1'):strict.update(GC_PROFILE_FOOTPRINT_FIELDS)
        fields[tf]={name:'gc_'+tf.lower()+'_'+name for name in sorted(observed|strict)}
    fields['strict_m5']={name:'gc_book_'+name for name in profile_dynamic_units()}
    return fields


def reference_profile_anchors(starts):
    """Independent IANA local-date arithmetic, distinct from producer pandas."""
    from datetime import timedelta, time
    from zoneinfo import ZoneInfo
    zone=ZoneInfo('America/New_York');cache={};out=[]
    for stamp in starts:
        local=datetime.fromtimestamp(int(stamp)//NS,timezone.utc).astimezone(zone)
        day=local.date()-timedelta(days=int(local.hour<18))
        if day not in cache:cache[day]=int(datetime.combine(day,time(18),tzinfo=zone).timestamp())*NS
        out.append(cache[day])
    return np.asarray(out,np.int64)


def _profile_summary(histogram):
    if not histogram or any(not isinstance(price,int) or price<=0 or not isinstance(qty,int) or qty<=0
            for price,qty in histogram.items()):raise RuntimeError('GC_PROFILE_REFERENCE_HISTOGRAM_DOMAIN')
    prices=sorted(histogram);volume=sum(histogram.values())
    if volume>2**53:raise RuntimeError('GC_PROFILE_REFERENCE_FLOAT_QUANTITY_BOUND')
    center=max(range(len(prices)),key=lambda i:(histogram[prices[i]],-prices[i]));left=right=center
    used=histogram[prices[center]]
    while used*10<volume*7:
        lower=(-1 if left==0 else histogram[prices[left-1]] if prices[left]-prices[left-1]==1 else 0)
        upper=(-1 if right==len(prices)-1 else histogram[prices[right+1]] if prices[right+1]-prices[right]==1 else 0)
        # Traversing empty ticks adds zero. If chosen, jump that empty run to
        # its next occupied level; the opposing immediate neighbor is unchanged.
        if lower>=upper:left-=1;used+=histogram[prices[left]]
        else:right+=1;used+=histogram[prices[right]]
    return {'vwap':(sum(price*histogram[price] for price in prices),volume),
        'poc':(prices[center],1),'val':(prices[left],1),'vah':(prices[right],1),'volume':volume}


def _profile_rational_price(numerator, denominator, tick_size, price_scale):
    from fractions import Fraction
    value=float(Fraction(int(numerator)*int(tick_size),int(denominator)*int(price_scale)))
    unit=np.finfo(np.float64).eps/2
    # Producer VWAP: integer division, tick-scale division, product; reference
    # rational conversion adds one rounding. Covers the fewer-op level prices.
    return value,4*unit*abs(value)/(1-4*unit)/(1-unit)


def reference_profile_histogram(histogram, tick_size, price_scale):
    """Python-int side totals, lower-crossing quantiles and contiguous 70% VA."""
    if (not histogram or any(not isinstance(price,int) or price<=0 or len(qty)!=3
            or any(not isinstance(v,int) or v<0 for v in qty) or sum(qty)<=0 for price,qty in histogram.items())):
        raise RuntimeError('GC_PROFILE_REFERENCE_HISTOGRAM_DOMAIN')
    prices=sorted(histogram);total={price:sum(histogram[price]) for price in prices};summary=_profile_summary(total)
    values={};bounds={}
    for name in ('vwap','poc','val','vah'):
        values[name+'_usd'],bounds[name+'_usd']=_profile_rational_price(*summary[name],tick_size,price_scale)
    values.update(volume_contracts=summary['volume'],price_levels=len(prices))
    cumulative=0;crossings={};thresholds={'q25_usd':(summary['volume']+3)//4,
        'median_usd':(summary['volume']+1)//2,'q75_usd':(summary['volume']*3+3)//4}
    for price in prices:
        cumulative+=total[price]
        for name,threshold in thresholds.items():
            if name not in crossings and cumulative>=threshold:crossings[name]=price
    for name,price in crossings.items():values[name],bounds[name]=_profile_rational_price(price,1,tick_size,price_scale)
    deltas=[histogram[price][0]-histogram[price][1] for price in prices]
    values.update(max_level_delta_contracts=max(deltas),min_level_delta_contracts=min(deltas))
    for side,index in (('buy',0),('sell',1)):
        winner=max(prices,key=lambda price:(histogram[price][index],-price))
        name=side+'_poc_usd'
        values[name],bounds[name]=(_profile_rational_price(winner,1,tick_size,price_scale)
            if histogram[winner][index] else (np.nan,0.))
    for name in values:bounds.setdefault(name,0.)
    return values,bounds


def _profile_distance(close_tick, level, level_value, level_error, tick_size, price_scale):
    from fractions import Fraction
    reference=float(Fraction(int(close_tick))-Fraction(*level))
    close,close_error=smc_price_reference(close_tick,tick_size,price_scale)
    tick,tick_error=smc_price_reference(1,tick_size,price_scale)
    unit=np.finfo(np.float64).eps/2
    numerator=close-level_value
    numerator_error=close_error+level_error+2*unit*(abs(close)+abs(level_value)+close_error+level_error)/(1-unit)
    rounded,bound=_event_book_ratio(numerator,tick,numerator_error,tick_error)
    # Triangle bound between the rational reference and independently rounded
    # operands; this is deterministic arithmetic, never a fitted tolerance.
    return reference,bound+abs(rounded-reference)


def reference_profile_step(state, start, histogram, contract, close_tick, anchor, accepted_minutes=5,
        *, tick_size, price_scale, valid=True):
    previous=state.get('previous');reset=previous is None or contract!=previous[1] or anchor!=previous[2]
    if previous is not None and contract==previous[1] and start<=previous[0]:
        raise RuntimeError('GC_PROFILE_REFERENCE_CONTRACT_STREAM_ORDER')
    if reset:state.update(session={},accepted=0)
    if reset or previous is None or not previous[3] or start-previous[0]!=5*MINUTE:state.update(rolling=[],segment={})
    units=profile_dynamic_units();values={name:False if unit=='binary' else np.nan for name,unit in units.items()}
    bounds={name:0. for name in values}
    values.update(session_accepted_minutes=0,session_expected_minutes=(start+5*MINUTE-anchor)//MINUTE)
    levels={}
    if not valid:state.update(rolling=[],segment={})
    else:
        if not histogram or not 0<accepted_minutes<=5:raise RuntimeError('GC_PROFILE_REFERENCE_VALID_WITHOUT_TRADES')
        state['rolling'].append(dict(histogram));state['rolling']=state['rolling'][-12:]
        for target in (state['segment'],state['session']):
            for price,qty in histogram.items():target[price]=target.get(price,0)+qty
        state['accepted']+=accepted_minutes
        families={'anchored_segment':state['segment'],'session_observed':state['session']}
        for length in (3,12):
            if len(state['rolling'])>=length:
                combined={}
                for hist in state['rolling'][-length:]:
                    for price,qty in hist.items():combined[price]=combined.get(price,0)+qty
                families['rolling'+str(length*5)]=combined
        for family,hist in families.items():
            summary=_profile_summary(hist);values[family+'_valid']=True;values[family+'_volume_contracts']=summary['volume']
            for level in ('vwap','poc','val','vah'):
                key=family+'_'+level;levels[key]=summary[level]
                values[key+'_usd'],bounds[key+'_usd']=_profile_rational_price(*summary[level],tick_size,price_scale)
        values['session_accepted_minutes']=state['accepted']
        values['session_complete_valid']=state['accepted']==values['session_expected_minutes']
        same=previous is not None and start-previous[0]==5*MINUTE and contract==previous[1] and previous[3]
        for family in GC_PROFILE_FAMILIES[:3]:
            for level in ('vwap','poc','val','vah'):
                key=family+'_'+level
                if key in levels:
                    values[key+'_distance_ticks'],bounds[key+'_distance_ticks']=_profile_distance(
                        close_tick,levels[key],values[key+'_usd'],bounds[key+'_usd'],tick_size,price_scale)
                if same and key in previous[4]:
                    values[key+'_prior_level_response_ticks'],bounds[key+'_prior_level_response_ticks']=_profile_distance(
                        close_tick,previous[4][key],previous[5][key+'_usd'],previous[6][key+'_usd'],tick_size,price_scale)
    state['previous']=(int(start),int(contract),int(anchor),bool(valid),levels,values,bounds)
    return values,bounds


def validate_profile_m1_schema(m1):
    if set(m1)!=set(GC_PROFILE_M1_COLUMNS):raise RuntimeError('GC_PROFILE_REFERENCE_M1_SCHEMA')
    minutes=np.asarray(m1['bar_start_ns']);shape=minutes.shape
    if (minutes.ndim!=1 or not len(minutes) or minutes.dtype!=np.dtype('int64') or np.any(np.diff(minutes)!=MINUTE)
            or any(np.asarray(value).shape!=shape for value in m1.values())
            or not np.issubdtype(np.asarray(m1['contract_index']).dtype,np.integer)
            or any(np.asarray(m1[name]).dtype.kind!='b' for name in ('volume_valid','signed_valid'))
            or any(np.asarray(m1[name]).dtype!=np.dtype('uint64') for name in ('volume','buy','sell','unknown'))
            or not np.issubdtype(np.asarray(m1['close_tick']).dtype,np.integer)):
        raise RuntimeError('GC_PROFILE_REFERENCE_M1_SCHEMA')
    admitted=m1['volume_valid']
    if (np.any(m1['signed_valid'] & ~admitted) or np.any(m1['signed_valid'] & (m1['unknown']!=0))
            or np.any(m1['contract_index'][admitted]<0)
            or np.any(m1['close_tick'][admitted]<=0) or np.any(m1['close_tick'][admitted]>2**53)):
        raise RuntimeError('GC_PROFILE_REFERENCE_M1_SCHEMA')


def reference_profile_grid(m1, timeframe):
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    validate_profile_m1_schema(m1)
    minutes=m1['bar_start_ns'];duration=MULTI_TF_SHIFT[timeframe].value;offset=TRADING_SESSION_BOUNDARY_OFFSET.value%duration
    labels=((minutes-offset)//duration)*duration+offset;stamps=np.unique(labels)
    starts=np.searchsorted(minutes,stamps);stops=np.searchsorted(minutes,stamps+duration);n=len(stamps)
    grid={'bar_start_ns':stamps,'available_ns':stamps+duration,'contract_index':np.zeros(n,np.int64)}
    grid.update({name:np.zeros(n,bool) for name in ('volume_valid','signed_valid','observed_valid')})
    grid.update({name:np.zeros(n,np.int64) for name in ('accepted_minutes','expected_minutes','observed_volume',
        'buy','sell','unknown','strict_close_tick','observed_close_tick')})
    for row,(stamp,begin,end) in enumerate(zip(stamps,starts,stops)):
        contracts=m1['contract_index'][begin:end];admitted=m1['volume_valid'][begin:end]
        same=bool(np.all(contracts>=0) and np.all(contracts==contracts[0]))
        full=(end-begin==duration//MINUTE and minutes[begin]==stamp and minutes[end-1]==stamp+duration-MINUTE)
        count=int(admitted.sum());grid['contract_index'][row]=contracts[0]
        grid['volume_valid'][row]=bool(full and same and admitted.all())
        grid['signed_valid'][row]=bool(grid['volume_valid'][row] and m1['signed_valid'][begin:end].all())
        grid['observed_valid'][row]=bool(full and same and count)
        grid['accepted_minutes'][row]=count;grid['expected_minutes'][row]=duration//MINUTE
        for side in ('buy','sell','unknown'):
            qty=sum(int(v) for v in m1[side][begin:end][admitted])
            if qty>2**53:raise RuntimeError('GC_PROFILE_REFERENCE_FLOAT_QUANTITY_BOUND')
            grid[side][row]=qty
        grid['observed_volume'][row]=sum(int(grid[side][row]) for side in ('buy','sell','unknown'))
        if grid['observed_volume'][row]>2**53:raise RuntimeError('GC_PROFILE_REFERENCE_FLOAT_QUANTITY_BOUND')
        grid['strict_close_tick'][row]=m1['close_tick'][end-1]
        if count:grid['observed_close_tick'][row]=m1['close_tick'][begin+np.flatnonzero(admitted)[-1]]
    return grid


def reference_profile_ladder(m1, batches, tick_size, price_scale):
    """One published ladder pass; five open TF maps, two M5 profile states.

    No all-M1 histogram collection is retained. TF outputs are caller-owned
    arrays, not a second source scan. Contract-major rows remain in stream order.
    """
    validate_profile_m1_schema(m1)
    tfs=('M5','M15','H1','H4','D1');grids={tf:reference_profile_grid(m1,tf) for tf in tfs}
    units=profile_reference_units();dynamic_units=profile_dynamic_units();references={};bounds={}
    for tf,grid in grids.items():
        n=len(grid['bar_start_ns']);references[tf]={name:np.full(n,False if unit=='binary' else np.nan)
            for name,unit in units.items()};bounds[tf]={name:np.zeros(n) for name in units}
        for name in ('bar_start_ns','available_ns','contract_index'):references[tf][name]=grid[name].copy()
    m5=grids['M5'];anchors=reference_profile_anchors(m5['bar_start_ns']);n=len(anchors)
    snapshots={mode:{name:np.full(n,False if unit=='binary' else np.nan) for name,unit in dynamic_units.items()}
        for mode in ('observed','strict')}
    snapshot_bounds={mode:{name:np.zeros(n) for name in dynamic_units} for mode in snapshots}
    for values in snapshots.values():
        values['session_accepted_minutes'][:]=0
        values['session_expected_minutes'][:]=(m5['available_ns']-anchors)//MINUTE
    states={tf:None for tf in tfs};emitted={tf:np.zeros(len(grid['bar_start_ns']),bool) for tf,grid in grids.items()}
    dynamic_states={'observed':{},'strict':{}};seen=np.zeros(len(m1['bar_start_ns']),bool)
    current=None;minute_hist={};minute_sides=[0,0,0];rows_seen=0;maximum_levels=0
    def flush_tf(tf):
        state=states[tf]
        if state is None:return
        row,hist=state;grid=grids[tf]
        if emitted[tf][row]:raise RuntimeError('GC_PROFILE_REFERENCE_TF_REVISIT')
        values,limits=reference_profile_histogram(hist,tick_size,price_scale)
        if values['volume_contracts']!=int(grid['observed_volume'][row]):
            raise RuntimeError('GC_PROFILE_REFERENCE_TF_VOLUME_CONSERVATION')
        for index,side in enumerate(('buy','sell','unknown')):
            if sum(qty[index] for qty in hist.values())!=int(grid[side][row]):
                raise RuntimeError('GC_PROFILE_REFERENCE_TF_SIDE_CONSERVATION')
        target=references[tf];errors=bounds[tf]
        if grid['observed_valid'][row]:
            for name in GC_PROFILE_HISTOGRAM_FIELDS+('val_usd','vah_usd'):
                target['observed_'+name][row]=values[name];errors['observed_'+name][row]=limits[name]
            for name,destination in GC_PROFILE_OBSERVED_FOOTPRINT.items():target[destination][row]=values[name];errors[destination][row]=limits[name]
        if grid['volume_valid'][row]:
            for name in GC_PROFILE_HISTOGRAM_FIELDS:target[name][row]=values[name];errors[name][row]=limits[name]
            if grid['signed_valid'][row]:
                for name in GC_PROFILE_FOOTPRINT_FIELDS:target[name][row]=values[name];errors[name][row]=limits[name]
        if tf=='M5':
            total={price:sum(qty) for price,qty in hist.items()}
            for mode,valid,close,accepted in (
                    ('observed',grid['observed_valid'][row],grid['observed_close_tick'][row],grid['accepted_minutes'][row]),
                    ('strict',grid['volume_valid'][row],grid['strict_close_tick'][row],5)):
                if valid:
                    result,error=reference_profile_step(dynamic_states[mode],int(grid['bar_start_ns'][row]),total,
                        int(grid['contract_index'][row]),int(close),int(anchors[row]),int(accepted),
                        tick_size=tick_size,price_scale=price_scale)
                    for name,value in result.items():snapshots[mode][name][row]=value;snapshot_bounds[mode][name][row]=error[name]
        emitted[tf][row]=True
    def flush_minute():
        nonlocal maximum_levels
        if current is None:return
        if seen[current]:raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_MINUTE_REVISIT')
        if (any(minute_sides[index]!=int(m1[side][current]) for index,side in enumerate(('buy','sell','unknown')))
                or sum(minute_sides)!=int(m1['volume'][current])):
            raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_SIDE_CONSERVATION')
        seen[current]=True;maximum_levels=max(maximum_levels,len(minute_hist))
        for tf,grid in grids.items():
            duration=int(grid['available_ns'][0]-grid['bar_start_ns'][0])
            row=(int(m1['bar_start_ns'][current])-int(grid['bar_start_ns'][0]))//duration
            if not grid['observed_valid'][row] and not grid['volume_valid'][row]:continue
            if states[tf] is None or states[tf][0]!=row:flush_tf(tf);states[tf]=(row,{})
            target=states[tf][1]
            for price,qty in minute_hist.items():
                old=target.get(price,(0,0,0));target[price]=tuple(old[j]+qty[j] for j in range(3))
    for batch in batches:
        if set(batch)!=set(GC_PROFILE_LADDER_COLUMNS):raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_SCHEMA')
        arrays={name:np.asarray(value) for name,value in batch.items()};clock=arrays['bar_start_ns'];shape=clock.shape
        if (clock.ndim!=1 or clock.dtype!=np.dtype('int64') or any(value.shape!=shape for value in arrays.values())
                or arrays['price_tick'].dtype!=np.dtype('int64') or not np.issubdtype(arrays['contract_index'].dtype,np.integer)
                or arrays['signed_valid'].dtype.kind!='b'
                or any(arrays[name].dtype!=np.dtype('uint64') for name in ('buy','sell','unknown'))):
            raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_SCHEMA')
        indices=np.searchsorted(m1['bar_start_ns'],clock);clipped=np.minimum(indices,len(seen)-1)
        if (np.any(indices>=len(seen)) or not np.array_equal(m1['bar_start_ns'][clipped],clock)
                or not np.array_equal(m1['contract_index'][clipped],arrays['contract_index'])
                or not m1['volume_valid'][clipped].all()
                or not np.array_equal(m1['signed_valid'][clipped],arrays['signed_valid'])
                or np.any(arrays['price_tick']<=0) or np.any(arrays['price_tick']>2**53)):
            raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_M1_MAPPING')
        for i,row in enumerate(indices):
            row=int(row)
            if current!=row:flush_minute();current=row;minute_hist={};minute_sides=[0,0,0]
            price=int(arrays['price_tick'][i])
            if price in minute_hist:raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_DUPLICATE_LEVEL')
            qty=tuple(int(arrays[side][i]) for side in ('buy','sell','unknown'))
            if sum(qty)<=0:raise RuntimeError('GC_PROFILE_REFERENCE_HISTOGRAM_DOMAIN')
            minute_hist[price]=qty
            for side in range(3):minute_sides[side]+=qty[side]
        rows_seen+=len(clock)
    flush_minute()
    for tf in tfs:
        flush_tf(tf)
        if not np.array_equal(emitted[tf],grids[tf]['observed_valid']):
            raise RuntimeError('GC_PROFILE_REFERENCE_TF_HISTOGRAM_COVERAGE')
    if not np.array_equal(seen,m1['volume_valid']):raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_MINUTE_COVERAGE')
    # Exact closed-M5 projection; missing snapshots are never backsearched.
    for tf,grid in grids.items():
        position=np.searchsorted(m5['available_ns'],grid['available_ns'],side='right')-1
        clip=np.clip(position,0,n-1);match=(position>=0)&(m5['available_ns'][clip]==grid['available_ns'])
        for name,values in snapshots['observed'].items():
            destination='observed_profile_'+name
            references[tf][destination]=match & values[clip].astype(bool) if dynamic_units[name]=='binary' else np.where(match,values[clip],np.nan)
            bounds[tf][destination]=snapshot_bounds['observed'][name][clip]
        references[tf]['observed_profile_source_coverage_fraction']=np.where(match,m5['accepted_minutes'][clip]/5,np.nan)
        # Division by5 has one rounding; both producer/reference share exact ints.
        bounds[tf]['observed_profile_source_coverage_fraction']=np.finfo(np.float64).eps/2*np.abs(references[tf]['observed_profile_source_coverage_fraction'])
        references[tf]['observed_profile_snapshot_available_ns']=np.where(match,m5['available_ns'][clip],np.nan)
        references[tf]['observed_profile_snapshot_age_minutes']=np.where(match,0.,np.nan)
    references['strict_m5']={name:m5[name].copy() for name in ('bar_start_ns','available_ns','contract_index')}
    references['strict_m5'].update(snapshots['strict']);bounds['strict_m5']={name:np.zeros(n) for name in references['strict_m5']}
    bounds['strict_m5'].update(snapshot_bounds['strict'])
    return references,bounds,{'ladder_rows':rows_seen,'admitted_minutes':int(m1['volume_valid'].sum()),
        'seen_minutes':int(seen.sum()),'ladder_passes':1,'maximum_single_minute_levels':maximum_levels,
        'open_timeframe_histograms':5,'dynamic_m5_domains':2,'all_m1_histograms_retained':False}


GC_BASIS_PLAN_ID = 'NATIVE_M5_GC_BASIS_NUMERIC_REFERENCE_20261021_125'
GC_BASIS_POLICY_KEY = 'gc_basis_numeric_reference_125'
GC_BASIS_M1_COLUMNS = ('bar_start_ns','contract_index','volume_valid','close_tick')
GC_BASIS_SPOT_COLUMNS = ('time','bid_close','ask_close')


def basis_numeric_active_fields():
    fields={tf:{'observed_spot_basis_'+name:'gc_'+tf.lower()+'_observed_spot_basis_'+name
        for name in ('bps','age_minutes')} for tf in ('M5','M15','H1','H4','D1')}
    fields.update({'strict_'+tf:{'spot_basis_bps':'gc_'+tf.lower()+'_spot_basis_bps'}
        for tf in ('M5','M15','H1','H4')})
    return fields


def validate_basis_m1_schema(m1):
    if set(m1)!=set(GC_BASIS_M1_COLUMNS):raise RuntimeError('GC_BASIS_REFERENCE_M1_SCHEMA')
    clock=np.asarray(m1['bar_start_ns']);shape=clock.shape
    if (clock.ndim!=1 or not len(clock) or clock.dtype!=np.dtype('int64') or np.any(np.diff(clock)!=MINUTE)
            or any(np.asarray(value).shape!=shape for value in m1.values())
            or not np.issubdtype(np.asarray(m1['contract_index']).dtype,np.integer)
            or np.asarray(m1['volume_valid']).dtype.kind!='b'
            or not np.issubdtype(np.asarray(m1['close_tick']).dtype,np.integer)):
        raise RuntimeError('GC_BASIS_REFERENCE_M1_SCHEMA')
    accepted=m1['volume_valid']
    if (np.any(m1['contract_index'][accepted]<0) or np.any(m1['close_tick'][accepted]<=0)
            or np.any(m1['close_tick'][accepted]>2**53)):
        raise RuntimeError('GC_BASIS_REFERENCE_M1_SCHEMA')


def reference_basis_spot(minutes, batches, seal):
    """One exact minute join; no older quote or endpoint substitution."""
    minutes=np.asarray(minutes)
    if (minutes.ndim!=1 or not len(minutes) or minutes.dtype!=np.dtype('int64')
            or np.any(np.diff(minutes)!=MINUTE)):
        raise RuntimeError('GC_BASIS_REFERENCE_M1_SCHEMA')
    spot={name:np.full(len(minutes),np.nan) for name in ('bid','ask')};seen=np.zeros(len(minutes),bool)
    rows=0;terminal_excluded=0
    for batch in batches:
        if set(batch)!=set(GC_BASIS_SPOT_COLUMNS):raise RuntimeError('GC_BASIS_REFERENCE_SPOT_SCHEMA')
        arrays={name:np.asarray(value) for name,value in batch.items()};clock=arrays['time'];shape=clock.shape
        if (clock.ndim!=1 or clock.dtype!=np.dtype('int64') or any(value.shape!=shape for value in arrays.values())
                or any(arrays[name].dtype!=np.dtype('float64') for name in ('bid_close','ask_close'))):
            raise RuntimeError('GC_BASIS_REFERENCE_SPOT_SCHEMA')
        if np.any(clock>=seal):raise RuntimeError('GC_BASIS_REFERENCE_SPOT_TEST_CLOCK')
        domain=(clock>=minutes[0])&(clock<minutes[-1]+MINUTE)
        terminal_excluded+=int(np.count_nonzero(domain&(clock+MINUTE>=seal)))
        take=domain&(clock+MINUTE<seal)
        times=clock[take];bid=arrays['bid_close'][take];ask=arrays['ask_close'][take]
        if (not np.isfinite(bid).all() or not np.isfinite(ask).all() or np.any(bid<=0) or np.any(ask<bid)):
            raise RuntimeError('GC_BASIS_REFERENCE_SPOT_QUOTES')
        if np.any(times%MINUTE):raise RuntimeError('GC_BASIS_REFERENCE_SPOT_TIME_GRID')
        positions=np.searchsorted(minutes,times)
        if np.any(positions>=len(minutes)) or not np.array_equal(minutes[positions],times):
            raise RuntimeError('GC_BASIS_REFERENCE_SPOT_TIME_GRID')
        if len(np.unique(positions))!=len(positions) or seen[positions].any():
            raise RuntimeError('GC_BASIS_REFERENCE_SPOT_DUPLICATE')
        seen[positions]=True;spot['bid'][positions]=bid;spot['ask'][positions]=ask;rows+=len(clock)
    return spot,{'decoded_spot_rows':rows,'matched_spot_minutes':int(seen.sum()),
        'missing_spot_minutes':int((~seen).sum()),'terminal_m1_rows_excluded':terminal_excluded,
        'spot_passes':1,'older_spot_fallbacks':0}


def _basis_rational_value(close_tick, bid, ask, tick_size, price_scale):
    """Exact rational interpretation of the published binary64 spot operands."""
    from fractions import Fraction
    source_mid=(float(bid)+float(ask))/2
    if not np.isfinite(source_mid) or source_mid<np.finfo(np.float64).tiny:
        raise RuntimeError('GC_BASIS_REFERENCE_UNSUPPORTED_FLOAT_ARITHMETIC')
    mid_exact=(Fraction.from_float(float(bid))+Fraction.from_float(float(ask)))/2
    gc_exact=Fraction(int(close_tick)*int(tick_size),int(price_scale))
    reference=float((gc_exact/mid_exact-1)*10000)
    gc,gc_error=smc_price_reference(close_tick,tick_size,price_scale)
    mid=float(mid_exact);unit=np.finfo(np.float64).eps/2
    # Spot add/2 and the independent rational conversion: gamma_2. Ordinary
    # positive prices avoid underflow; division by2 is exact but need not be.
    mid_error=2*unit*abs(mid)/(1-2*unit)/(1-unit)
    ratio,ratio_error=_event_book_ratio(gc,mid,gc_error,mid_error)
    rounded=(ratio-1)*10000
    # Both subtraction/product paths are included, including cancellation near
    # zero basis. An absolute operand bound replaces a relative-only tolerance.
    subtract_error=2*unit*(abs(ratio)+1+ratio_error)/(1-unit)
    bound=10000*(ratio_error+subtract_error)/(1-unit)+2*unit*abs(rounded)/(1-unit)
    return reference,bound+abs(rounded-reference)


def reference_basis_aggregates(m1, spot, timeframe, tick_size, price_scale):
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.time.session_detector import TRADING_SESSION_BOUNDARY_OFFSET
    validate_basis_m1_schema(m1)
    clock=m1['bar_start_ns'];n=len(clock)
    if (set(spot)!=set(('bid','ask')) or any(np.asarray(value).shape!=(n,) or np.asarray(value).dtype!=np.dtype('float64')
            for value in spot.values()) or not np.array_equal(np.isnan(spot['bid']),np.isnan(spot['ask']))
            or any(np.isinf(value).any() for value in spot.values())):
        raise RuntimeError('GC_BASIS_REFERENCE_SPOT_SCHEMA')
    finite=np.isfinite(spot['bid'])
    if np.any(spot['bid'][finite]<=0) or np.any(spot['ask'][finite]<spot['bid'][finite]):
        raise RuntimeError('GC_BASIS_REFERENCE_SPOT_QUOTES')
    duration=MULTI_TF_SHIFT[timeframe].value;offset=TRADING_SESSION_BOUNDARY_OFFSET.value%duration
    labels=((clock-offset)//duration)*duration+offset;stamps=np.unique(labels)
    first=np.searchsorted(clock,stamps);last=np.searchsorted(clock,stamps+duration)
    common={'bar_start_ns':stamps,'available_ns':stamps+duration,'contract_index':np.zeros(len(stamps),np.int64)}
    observed={name:value.copy() for name,value in common.items()};strict={name:value.copy() for name,value in common.items()}
    observed.update({name:np.full(len(stamps),np.nan) for name in ('observed_spot_basis_bps',
        'observed_spot_basis_source_available_ns','observed_spot_basis_age_minutes')})
    observed['observed_spot_basis_valid']=np.zeros(len(stamps),bool)
    strict.update(volume_valid=np.zeros(len(stamps),bool),close_usd=np.full(len(stamps),np.nan),spot_basis_bps=np.full(len(stamps),np.nan))
    units={'observed':{name:'utc_nanoseconds_metadata' for name in common},'strict':{name:'utc_nanoseconds_metadata' for name in common}}
    for values in units.values():values['contract_index']='contract_index_metadata'
    units['observed'].update(observed_spot_basis_bps='bps',observed_spot_basis_valid='binary',
        observed_spot_basis_source_available_ns='utc_nanoseconds_metadata',observed_spot_basis_age_minutes='minutes')
    units['strict'].update(volume_valid='binary',close_usd='USD',spot_basis_bps='bps')
    bounds={'observed':{name:np.zeros(len(stamps)) for name in observed},'strict':{name:np.zeros(len(stamps)) for name in strict}}
    for row,(stamp,begin,end) in enumerate(zip(stamps,first,last)):
        contracts=m1['contract_index'][begin:end];accepted=m1['volume_valid'][begin:end]
        same=bool(np.all(contracts>=0) and np.all(contracts==contracts[0]))
        full=(end-begin==duration//MINUTE and clock[begin]==stamp and clock[end-1]==stamp+duration-MINUTE)
        strict['contract_index'][row]=observed['contract_index'][row]=contracts[0]
        strict_valid=bool(full and same and accepted.all());strict['volume_valid'][row]=strict_valid
        if strict_valid:
            value,error=smc_price_reference(int(m1['close_tick'][end-1]),tick_size,price_scale)
            strict['close_usd'][row]=value;bounds['strict']['close_usd'][row]=error
            if finite[end-1]:
                strict['spot_basis_bps'][row],bounds['strict']['spot_basis_bps'][row]=_basis_rational_value(
                    m1['close_tick'][end-1],spot['bid'][end-1],spot['ask'][end-1],tick_size,price_scale)
        if full and same and accepted.any():
            position=begin+np.flatnonzero(accepted)[-1]
            if finite[position]:
                value,error=_basis_rational_value(m1['close_tick'][position],spot['bid'][position],spot['ask'][position],tick_size,price_scale)
                source_clock=int(clock[position])+MINUTE
                if int(float(source_clock))!=source_clock:raise RuntimeError('GC_BASIS_REFERENCE_CLOCK_PRECISION')
                observed['observed_spot_basis_valid'][row]=True;observed['observed_spot_basis_bps'][row]=value
                bounds['observed']['observed_spot_basis_bps'][row]=error
                observed['observed_spot_basis_source_available_ns'][row]=source_clock
                observed['observed_spot_basis_age_minutes'][row]=(int(stamp)+duration-source_clock)//MINUTE
    return observed,strict,bounds,units


def read_basis_numeric_inputs(inputs, start, end, seal):
    m1,groups=read_event_numeric_columns(Path(inputs['m1_basis']['path']),list(GC_BASIS_M1_COLUMNS),start,end,seal)
    validate_basis_m1_schema(m1)
    return m1,groups


def iter_basis_spot(path, start, end, seal):
    parquet,groups=event_numeric_row_groups(path,start,end,seal,clock_column='time')
    if (not set(GC_BASIS_SPOT_COLUMNS)<=set(parquet.schema_arrow.names)
            or parquet.schema_arrow.field('time').type!=pa.timestamp('ns',tz='UTC')):
        raise RuntimeError('GC_BASIS_REFERENCE_SPOT_SCHEMA')
    def stream():
        for batch in parquet.iter_batches(batch_size=65536,row_groups=groups,columns=list(GC_BASIS_SPOT_COLUMNS),use_threads=False):
            values={name:batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False) for name in GC_BASIS_SPOT_COLUMNS}
            values['time']=pd.DatetimeIndex(values['time']).as_unit('ns').asi8
            yield values
    return stream(),groups


def event_numeric_row_groups(path, start, end, seal, *, clock_column='bar_start_ns'):
    """Inspect all selected footer boundaries before decoding any payload."""
    parquet = pq.ParquetFile(path)
    if clock_column=='time' and (not set(GC_BASIS_SPOT_COLUMNS)<=set(parquet.schema_arrow.names)
            or parquet.schema_arrow.field('time').type!=pa.timestamp('ns',tz='UTC')):
        raise RuntimeError('GC_BASIS_REFERENCE_SPOT_SCHEMA')
    clock = parquet.schema_arrow.names.index(clock_column)
    groups = []
    for row in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row).column(clock).statistics
        if stats is None or not stats.has_min_max:
            raise RuntimeError('GC_EVENT_NUMERIC_ROW_GROUP_CLOCK_UNPROVEN')
        low,high=([pd.Timestamp(value).value for value in (stats.min,stats.max)] if clock_column=='time'
            else (int(stats.min),int(stats.max)))
        if high < start or low >= end:
            continue
        if high >= seal:
            raise RuntimeError('GC_EVENT_NUMERIC_ROW_GROUP_INTERSECTS_TEST')
        groups.append(row)
    if not groups:
        raise RuntimeError('GC_EVENT_NUMERIC_NO_SAFE_ROW_GROUPS')
    return parquet,groups


def read_event_numeric_columns(path, columns, start, end, seal):
    parquet,groups=event_numeric_row_groups(path,start,end,seal)
    table = parquet.read_row_groups(groups, columns=columns, use_threads=False)
    times = table['bar_start_ns'].to_numpy()
    if np.any(times >= seal):
        raise RuntimeError('GC_EVENT_NUMERIC_TEST_CLOCK')
    take = (times >= start) & (times < end)
    values = {name: table[name].to_numpy()[take] for name in columns}
    return values, groups


def read_profile_numeric_inputs(inputs, start, end, seal):
    m1,groups=read_event_numeric_columns(Path(inputs['m1_profile']['path']),list(GC_PROFILE_M1_COLUMNS),start,end,seal)
    validate_profile_m1_schema(m1)
    return m1,groups


def iter_profile_numeric_ladder(path, start, end, seal):
    parquet,groups=event_numeric_row_groups(path,start,end,seal)
    if set(parquet.schema_arrow.names)!=set(GC_PROFILE_LADDER_COLUMNS):
        raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_SCHEMA')
    def stream():
        for batch in parquet.iter_batches(batch_size=65536,row_groups=groups,columns=list(GC_PROFILE_LADDER_COLUMNS),use_threads=False):
            values={name:batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False) for name in GC_PROFILE_LADDER_COLUMNS}
            clock=values['bar_start_ns']
            if np.any(clock>=seal):raise RuntimeError('GC_PROFILE_REFERENCE_LADDER_TEST_CLOCK')
            take=(clock>=start)&(clock<end)
            yield {name:value[take] for name,value in values.items()}
    return stream(),groups


def read_event_numeric_book(path, start, end, seal, events):
    """The published book M1 has clock/contract keys, no availability column."""
    common = ['bar_start_ns','contract_index']
    columns = common+['ofi_valid','bbo_valid','order_count_valid','covered_ns','order_covered_ns',
        'opening_mid_tick','closing_mid_tick']+list(GC_BOOK_REFERENCE_ADDITIVE)+list(GC_BOOK_REFERENCE_MEANS)+list(GC_ORDER_REFERENCE_MEANS)
    book,groups=read_event_numeric_columns(path,columns,start,end,seal)
    if any(not np.array_equal(events[name],book[name]) for name in common):
        raise RuntimeError('GC_EVENT_BOOK_SHARED_M1_CLOCK_OR_CONTRACT')
    return book,groups


def read_trade_numeric_inputs(inputs, start, end, seal):
    core=['bar_start_ns','contract_index','volume_valid','signed_valid']+list(GC_TRADE_QUANTITIES)
    trade,groups=read_event_numeric_columns(Path(inputs['m1_trade']['path']),core,start,end,seal)
    sizes,size_groups=read_event_numeric_columns(Path(inputs['m1_trade_sizes']['path']),
        ['bar_start_ns','available_ns','contract_index']+list(GC_TRADE_SIZE_STATISTICS),start,end,seal)
    if (any(np.asarray(value).dtype!=np.dtype('int64') for value in
            (trade['bar_start_ns'],sizes['bar_start_ns'],sizes['available_ns']))
            or any(not np.issubdtype(np.asarray(table['contract_index']).dtype,np.integer) for table in (trade,sizes))
            or any(not np.array_equal(trade[name],sizes[name]) for name in ('bar_start_ns','contract_index'))
            or not np.array_equal(sizes['available_ns'],sizes['bar_start_ns']+MINUTE)):
        raise RuntimeError('GC_TRADE_REFERENCE_PUBLISHED_M1_JOIN_OR_AVAILABILITY')
    return trade,{name:sizes[name] for name in GC_TRADE_SIZE_STATISTICS},groups,size_groups


def run_gc_event_numeric(binding_path, *, event_book=False, trade=False, smc=False, profile=False, basis=False):
    plan_id = GC_BASIS_PLAN_ID if basis else GC_PROFILE_PLAN_ID if profile else GC_SMC_PLAN_ID if smc else GC_TRADE_PLAN_ID if trade else GC_EVENT_BOOK_PLAN_ID if event_book else GC_EVENT_NUMERIC_PLAN_ID
    policy_key = GC_BASIS_POLICY_KEY if basis else GC_PROFILE_POLICY_KEY if profile else GC_SMC_POLICY_KEY if smc else GC_TRADE_POLICY_KEY if trade else GC_EVENT_BOOK_POLICY_KEY if event_book else GC_EVENT_NUMERIC_POLICY_KEY
    root = Path('/home/andre2/GX1_RUNS')/plan_id
    prefix = 'GC_BASIS_NUMERIC' if basis else 'GC_PROFILE_NUMERIC' if profile else 'GC_SMC_NUMERIC' if smc else 'GC_TRADE_NUMERIC' if trade else 'GC_EVENT_BOOK_NUMERIC' if event_book else 'GC_EVENT_NUMERIC'
    schema = 'gx1_gc_basis_numeric_audit_plan_v1' if basis else 'gx1_gc_profile_numeric_audit_plan_v1' if profile else 'gx1_gc_smc_numeric_audit_plan_v1' if smc else 'gx1_gc_trade_numeric_audit_plan_v1' if trade else 'gx1_gc_event_book_numeric_audit_plan_v1' if event_book else 'gx1_gc_event_numeric_audit_plan_v1'
    if binding_path != root/(prefix+'_EXECUTION_BINDING.json'):
        raise RuntimeError('GC_EVENT_NUMERIC_BINDING_PATH')
    io.require_regular_file(binding_path, 'GC_EVENT_NUMERIC_BINDING')
    binding = io.read_json(binding_path); plan_path = root/'PLAN.json'
    io.verify_digest(plan_path, binding['plan_sha256'], 'GC_EVENT_NUMERIC_PLAN'); plan = io.read_json(plan_path)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'GC_EVENT_NUMERIC_POLICY')
    policy = io.read_json(REPO/'NEXT_RUN_POLICY.json'); gate = policy['current_work'][policy_key]
    if (plan['schema_version'] != schema
            or plan['plan_id'] != plan_id or plan['run_root'] != str(root)
            or plan['policy_key'] != policy_key or plan['max_invocations'] != 1
            or gate['status'] != 'EXECUTION_CLAIMED' or gate['claimed_invocations'] != 1
            or gate['completed_invocations'] != 0 or gate['execution_authorized'] is not False
            or gate['relaunch_allowed'] is not False
            or gate['plan'] != {'path': str(plan_path), 'sha256': binding['plan_sha256']}
            or policy['training_enabled'] is not False or io.git_head() != binding['source_head']
            or subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True)):
        raise RuntimeError('GC_EVENT_NUMERIC_AUTHORITY')
    if set(binding['code_sha256']) != set(plan['code_paths']):
        raise RuntimeError('GC_EVENT_NUMERIC_CODE_INVENTORY')
    for name, digest in binding['code_sha256'].items(): io.verify_digest(REPO/name, digest, 'GC_EVENT_NUMERIC_CODE')
    proof = require_capped_cpu_audit_execution(); proof['lock'] = require_capped_lock_ancestry()
    io.write_json(root/(prefix+'_CLAIM.json'), {'invocation': 1, 'plan_sha256': binding['plan_sha256'],
        'source_head': binding['source_head']})
    staging = root/('.'+prefix+'_RESULTS.staging'); staging.mkdir(exist_ok=False)
    stamps = [pd.Timestamp(plan[name]) for name in ('scan_start_utc','scan_end_utc','TEST_seal_utc')]
    if any(t.tzinfo is None for t in stamps): raise RuntimeError('GC_EVENT_NUMERIC_TIMEZONE')
    start, end, seal = [t.value for t in stamps]
    from gx1.contracts.gc_source_disposition_v1 import STUDY_START_NS, WINDOW_END_NS
    if (not STUDY_START_NS <= start < end <= seal or seal != WINDOW_END_NS
            or ((trade or smc or profile or basis) and start != STUDY_START_NS)):
        raise RuntimeError('GC_EVENT_NUMERIC_SCAN_BOUNDARY')
    publication = binding_read(plan['publication_terminal'])
    source_root = Path(plan['publication_terminal']['path']).parent
    if (source_root != Path('/home/andre2/GX1_RUNS/NATIVE_GC_TRUSTED_RECORD_RESTORATION_20261018_024/RESULTS')
            or Path(plan['publication_terminal']['path']).name != 'TERMINAL.json'
            or publication['status'] != 'COMPLETE' or publication['exit_code'] != 0
            or publication['source_unchanged'] is not True or publication['test_data_used'] is not False):
        raise RuntimeError('GC_EVENT_NUMERIC_PUBLICATION')
    inputs = plan['inputs']; tfs = ('M5','M15','H1','H4','D1')
    keys = {'m1_basis','roll_schedule','spot_manifest','spot_partitions','timeframes','strict_basis'} if basis else {'m1_profile','m1_ladder','roll_schedule','timeframes','strict_m5_profile'} if profile else {'m1_smc','roll_schedule','timeframes','strict_m5_price'} if smc else {'m1_trade','m1_trade_sizes','timeframes'} if trade else {'m1_events','m1_book','strict_m5_book','timeframes'} if event_book else {'m1_events','timeframes'}
    if set(inputs) != keys or set(inputs['timeframes']) != set(tfs):
        raise RuntimeError('GC_EVENT_NUMERIC_INPUT_INVENTORY')
    files = {'M1_GC_INPUTS.parquet':inputs['m1_basis'],'ROLL_SCHEDULE.json':inputs['roll_schedule'],
        'RESULT.json':plan['publication_result']} if basis else {'M1_GC_INPUTS.parquet':inputs['m1_profile'],'FOOTPRINT_M1_LEVELS.parquet':inputs['m1_ladder'],
        'ROLL_SCHEDULE.json':inputs['roll_schedule'],'M5_GC_FEATURES.parquet':inputs['strict_m5_profile']} if profile else {'M1_GC_INPUTS.parquet':inputs['m1_smc'],'ROLL_SCHEDULE.json':inputs['roll_schedule'],
        'M5_GC_FEATURES.parquet':inputs['strict_m5_price']} if smc else {'M1_GC_INPUTS.parquet':inputs['m1_trade'],
        'M1_TRADE_SIZE_STATISTICS_UNMASKED.parquet':inputs['m1_trade_sizes']} if trade else {'M1_EVENT_DIAGNOSTICS.parquet': inputs['m1_events']}
    files.update({'TRADE_FEATURES_'+tf+'.parquet': inputs['timeframes'][tf] for tf in tfs})
    if basis:
        if set(inputs['strict_basis'])!=set(tfs) or set(inputs['spot_partitions'])!={'2024','2025','2026'}:
            raise RuntimeError('GC_BASIS_REFERENCE_INPUT_INVENTORY')
        files.update({'BASIS_'+tf+'.parquet':inputs['strict_basis'][tf] for tf in tfs})
    if event_book:
        files.update({'M1_GC_INPUTS.parquet':inputs['m1_book'],'M5_GC_FEATURES.parquet':inputs['strict_m5_book']})
    identities = {}
    for name, item in files.items():
        path = source_root/name; artifact = publication['artifacts'][name]
        if item != {'path': str(path), 'sha256': artifact['sha256']}:
            raise RuntimeError('GC_EVENT_NUMERIC_PUBLISHED_BINDING:' + name)
        before = path.stat(); binding_read(item); after = path.stat()
        identity = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(before) != identity(after) or after.st_size != artifact['bytes']:
            raise RuntimeError('GC_EVENT_NUMERIC_INPUT_CHANGED:' + name)
        identities[str(path)] = identity(after)
    if basis:
        external=[plan['source_basis_plan'],plan['pending_field_inventory'],inputs['spot_manifest'],*inputs['spot_partitions'].values()]
        for item in external:
            path=Path(item['path']);io.require_regular_file(path,'GC_BASIS_REFERENCE_EXTERNAL_INPUT')
            if not path.is_absolute() or path!=path.resolve():raise RuntimeError('GC_BASIS_REFERENCE_CANONICAL_INPUT')
            before=path.stat();binding_read(item);after=path.stat()
            if identity(before)!=identity(after):raise RuntimeError('GC_BASIS_REFERENCE_EXTERNAL_CHANGED')
            identities[str(path)]=identity(after)
        source_plan=io.read_json(Path(plan['source_basis_plan']['path']))
        restored=io.read_json(Path(plan['publication_result']['path']));manifest=io.read_json(Path(inputs['spot_manifest']['path']))
        source_spot=source_plan['inputs']['oanda_manifest'];spot_root=Path(inputs['spot_manifest']['path']).parent
        parts=[inputs['spot_partitions'][year] for year in ('2024','2025','2026')]
        provenance={item['path']:item['sha256'] for item in restored['spot_basis_sources']}
        expected={item['path']:item['sha256'] for item in [plan['source_basis_plan'],inputs['spot_manifest'],*parts]}
        if (provenance!=expected or {name:source_spot[name] for name in ('path','sha256')}!=inputs['spot_manifest']
                or source_spot['partitions']!=parts or source_spot['read_only_columns']!=list(GC_BASIS_SPOT_COLUMNS)
                or source_spot['field']!='time' or source_spot['semantics']!='bar_start_utc'
                or source_spot['availability_offset_seconds']!=60
                or manifest['schema_version']!='xau_canonical_native_source_v3' or manifest['instrument']!='XAU_USD'
                or manifest['timeframe']!='M1' or manifest['prices']!='MBA' or manifest['completion_field']!='complete'
                or manifest['completion_value'] is not True or manifest['bar_duration_seconds']!=60
                or manifest['decision_available_offset_seconds']!=60 or manifest['timestamp_semantics']!='bar_start_utc'
                or pd.Timestamp(manifest['requested_end_utc_exclusive']).value!=seal or manifest['out_root']!=str(spot_root)
                or any(item!={'path':str(spot_root/('year='+year)/'part-000.parquet'),
                    'sha256':manifest['year_sha256']['year='+year]} for year,item in inputs['spot_partitions'].items())):
            raise RuntimeError('GC_BASIS_REFERENCE_SPOT_PROVENANCE')
        pending=io.read_json(Path(plan['pending_field_inventory']['path']))['groups']['spot_futures_basis']
        active={name for fields in basis_numeric_active_fields().values() for name in fields.values()}
        if len(active)!=14 or len(pending)!=14 or {item['name'] for item in pending}!=active:
            raise RuntimeError('GC_BASIS_REFERENCE_PENDING_14_INVENTORY')
    common = ['bar_start_ns','available_ns','contract_index']
    source_columns = common + ['qualified_final_event_count'] + [d+s for d in GC_EVENT_CLOCK_DOMAINS for s in ('_count','_sum_ns')]
    if event_book:source_columns=common+list(GC_EVENT_REFERENCE_REDUCERS)+['qualified_covered_ns']
    if profile or basis:
        # All selected GC/spot footer barriers precede ANY decoded payload.
        for name,item in files.items():
            if name.endswith('.parquet'):event_numeric_row_groups(Path(item['path']),start,end,seal)
        if basis:
            for item in inputs['spot_partitions'].values():event_numeric_row_groups(Path(item['path']),start,end,seal,clock_column='time')
    if profile or smc or basis:
        m1,m1_groups=(read_basis_numeric_inputs(inputs,start,end,seal) if basis else
            read_profile_numeric_inputs(inputs,start,end,seal) if profile else read_smc_numeric_inputs(inputs,start,end,seal))
        stamps=m1['bar_start_ns']
        if (stamps.dtype!=np.dtype('int64') or not len(stamps) or stamps[0]!=start or stamps[-1]!=end-MINUTE
                or len(stamps)!=(end-start)//MINUTE or np.any(np.diff(stamps)!=MINUTE)):
            raise RuntimeError('GC_SMC_REFERENCE_FULL_PUBLISHED_M1_GRID')
        from gx1.features.gc_order_flow_v1 import PRICE_SCALE
        roll=io.read_json(Path(inputs['roll_schedule']['path']))
        symbols=roll['contract_order'];metadata=roll['metadata_identity']
        ticks={metadata[symbol][2] for symbol in symbols}
        if (roll['origin_ns']!=start or not symbols or len(set(symbols))!=len(symbols) or len(ticks)!=1
                or not np.issubdtype(m1['contract_index'].dtype,np.integer)
                or np.any(m1['contract_index']>=len(symbols))):
            raise RuntimeError('GC_SMC_REFERENCE_ROLL_METADATA')
        tick_size=next(iter(ticks))
        if not isinstance(tick_size,int) or tick_size<=0:raise RuntimeError('GC_PROFILE_REFERENCE_TICK_METADATA')
        if basis:
            from itertools import chain
            readers=[iter_basis_spot(Path(inputs['spot_partitions'][year]['path']),start,end,seal) for year in ('2024','2025','2026')]
            spot,spot_report=reference_basis_spot(m1['bar_start_ns'],chain.from_iterable(item[0] for item in readers),seal)
            spot_groups={year:readers[i][1] for i,year in enumerate(('2024','2025','2026'))}
        elif profile:
            inventory=binding_read(plan['pending_field_inventory']);declared=profile_numeric_active_fields()
            expected={name for fields in declared.values() for name in fields.values()}
            pending=[item['name'] for group in ('dynamic_profiles','native_timeframe_histogram','footprint')
                for item in inventory['groups'][group]]
            if len(expected)!=412 or len(pending)!=412 or set(pending)!=expected:
                raise RuntimeError('GC_PROFILE_REFERENCE_PENDING_412_INVENTORY')
            batches,ladder_groups=iter_profile_numeric_ladder(Path(inputs['m1_ladder']['path']),start,end,seal)
            profile_refs,profile_bounds,profile_report=reference_profile_ladder(m1,batches,tick_size,PRICE_SCALE)
        else:
            tolerance=plan['pivot_comparison_tolerance_usd']
            if tolerance!=1e-12:raise RuntimeError('GC_SMC_REFERENCE_OWNER_PIVOT_TOLERANCE_CHANGED')
            accepted=np.asarray(m1['volume_valid'],bool)
            maximum=max(int(np.max(np.abs(m1[name][accepted]),initial=0)) for name in GC_SMC_M1_COLUMNS[4:])
            price_order=validate_smc_tick_order(maximum,tick_size,PRICE_SCALE,tolerance)
    elif trade:
        m1,trade_stats,m1_groups,trade_size_groups=read_trade_numeric_inputs(inputs,start,end,seal)
        stamps=m1['bar_start_ns']
        if (not len(stamps) or stamps[0]!=start or stamps[-1]!=end-MINUTE
                or len(stamps)!=(end-start)//MINUTE or np.any(np.diff(stamps)!=MINUTE)):
            raise RuntimeError('GC_TRADE_REFERENCE_FULL_PUBLISHED_M1_GRID')
        known,cvd=reference_trade_cumulatives(stamps,m1['contract_index'],m1,trade_stats)
    else:
        m1, m1_groups = read_event_numeric_columns(Path(inputs['m1_events']['path']),source_columns,start,end,seal)
        if not np.array_equal(m1['available_ns'],m1['bar_start_ns']+MINUTE):
            raise RuntimeError('GC_EVENT_NUMERIC_M1_AVAILABILITY')
    reports = {}; strict_report = None; book_groups = None
    if event_book:
        book,book_groups=read_event_numeric_book(Path(inputs['m1_book']['path']),start,end,seal,m1)
    for tf in tfs:
        if basis:
            reference,basis_strict,scoped_bounds,scoped_units=reference_basis_aggregates(m1,spot,tf,tick_size,PRICE_SCALE)
            bounds,units=scoped_bounds['observed'],scoped_units['observed']
        elif profile:
            reference,bounds,units=profile_refs.pop(tf),profile_bounds.pop(tf),profile_reference_units()
        elif smc:
            reference,bounds,units=reference_smc_aggregates(m1,tf,tick_size,PRICE_SCALE)
        elif trade:
            reference,bounds,units=reference_trade_aggregates(
                m1['bar_start_ns'],m1['contract_index'],m1,trade_stats,known,cvd,tf)
        elif event_book:
            reference,bounds,units,clock,strict=reference_event_book_aggregates(
                m1['bar_start_ns'],m1['contract_index'],m1,book,tf)
        else:reference = reference_event_clock_means(m1['bar_start_ns'],m1['contract_index'],m1,tf)
        keep = (reference['bar_start_ns'] >= start) & (reference['available_ns'] <= end)
        excluded = int((~keep).sum()); reference_keep=keep
        reference = {name: value[keep] for name,value in reference.items()}
        columns = common[:2] + ['observed_event_valid','observed_event_accepted_minutes'] + [
            'observed_event_'+d+s for d in GC_EVENT_CLOCK_DOMAINS for s in ('_count','_sum_ns','_mean_ns','_valid')]
        if event_book or trade or smc or profile or basis:columns=list(reference)
        actual, groups = read_event_numeric_columns(Path(inputs['timeframes'][tf]['path']),columns,start,end,seal)
        keep = actual['available_ns'] <= end; actual = {name:value[keep] for name,value in actual.items()}
        if basis:
            bounds={name:value[reference_keep] for name,value in bounds.items()}
            reports[tf]={'fields':compare_trade_reference(reference,actual,bounds,units)}
            reports[tf]['fields']['observed_spot_basis_valid']['positive_reference_values']=int(np.count_nonzero(reference['observed_spot_basis_valid']))
            values={name:value[reference_keep] for name,value in basis_strict.items()}
            limits={name:value[reference_keep] for name,value in scoped_bounds['strict'].items()}
            old,old_groups=read_event_numeric_columns(Path(inputs['strict_basis'][tf]['path']),list(values),start,end,seal)
            take=old['available_ns']<=end;old={name:value[take] for name,value in old.items()}
            reports[tf]['strict_fields']=compare_trade_reference(values,old,limits,scoped_units['strict'])
            reports[tf]['strict_decoded_row_groups']=old_groups
        elif profile:
            bounds={name:value[reference_keep] for name,value in bounds.items()}
            reports[tf]={'fields':compare_trade_reference(reference,actual,bounds,units)}
            for name,item in reports[tf]['fields'].items():
                if units[name]=='binary':item['positive_reference_values']=int(np.count_nonzero(reference[name]==1))
            if tf=='M5':
                values=profile_refs.pop('strict_m5');limits=profile_bounds.pop('strict_m5')
                rows=(values['bar_start_ns']>=start)&(values['available_ns']<=end)
                values={name:value[rows] for name,value in values.items()};limits={name:value[rows] for name,value in limits.items()}
                strict_units={name:profile_reference_units()[name] for name in common};strict_units.update(profile_dynamic_units())
                old,old_groups=read_event_numeric_columns(Path(inputs['strict_m5_profile']['path']),list(values),start,end,seal)
                take=old['available_ns']<=end;old={name:value[take] for name,value in old.items()}
                strict_report={'fields':compare_trade_reference(values,old,limits,strict_units),'decoded_row_groups':old_groups}
                for name,item in strict_report['fields'].items():
                    if strict_units[name]=='binary':item['positive_reference_values']=int(np.count_nonzero(values[name]==1))
        elif smc:
            bounds={name:value[reference_keep] for name,value in bounds.items()}
            reports[tf]={'fields':compare_trade_reference(reference,actual,bounds,units)}
            for name,item in reports[tf]['fields'].items():
                if units[name]=='binary':item['positive_reference_values']=int(np.count_nonzero(reference[name]==1))
            if tf=='M5':
                names=common+['close_usd'];values={name:reference[name] for name in names}
                limits={name:bounds[name] for name in names};price_units={name:units[name] for name in names}
                old,old_groups=read_event_numeric_columns(Path(inputs['strict_m5_price']['path']),names,start,end,seal)
                take=old['available_ns']<=end;old={name:value[take] for name,value in old.items()}
                strict_report={'fields':compare_trade_reference(values,old,limits,price_units),'decoded_row_groups':old_groups}
        elif trade:
            bounds={name:value[reference_keep] for name,value in bounds.items()}
            reports[tf]={'fields':compare_trade_reference(reference,actual,bounds,units),
                'known_endpoint_before_tf_end_buckets':int(np.count_nonzero(
                    reference['observed_valid'] & (reference['observed_last_minute_ns']<reference['available_ns']-MINUTE)))}
        elif event_book:
            bounds={name:value[reference_keep] for name,value in bounds.items()}
            clock={name:value[reference_keep] for name,value in clock.items()}
            reports[tf]={'clock_mean_discrimination':compare_event_clock_reference(clock,actual),
                'fields':compare_event_book_reference(reference,actual,bounds,units)}
            if tf=='M5':
                values,limits,strict_units=strict
                # M5 scan boundaries are checked against the independently built grid.
                rows=reference_keep
                values={name:value[rows] for name,value in values.items()}
                limits={name:value[rows] for name,value in limits.items()}
                old,old_groups=read_event_numeric_columns(Path(inputs['strict_m5_book']['path']),
                    common[:2]+list(values),start,end,seal)
                take=old['available_ns']<=end;old={name:value[take] for name,value in old.items()}
                if any(not np.array_equal(reference[name],old.pop(name)) for name in common[:2]):
                    raise RuntimeError('GC_EVENT_BOOK_STRICT_GRID')
                strict_report={'fields':compare_event_book_reference(values,old,limits,strict_units),
                    'decoded_row_groups':old_groups}
        else:reports[tf] = compare_event_clock_reference(reference,actual)
        reports[tf]['decoded_row_groups'] = groups
        reports[tf]['partial_scan_edge_buckets_excluded'] = excluded
    for path, wanted in identities.items():
        io.require_regular_file(Path(path),'GC_EVENT_NUMERIC_POST_INPUT')
        if identity(Path(path).stat()) != wanted: raise RuntimeError('GC_EVENT_NUMERIC_POST_INPUT_CHANGED')
    for name,digest in binding['code_sha256'].items(): io.verify_digest(REPO/name,digest,'GC_EVENT_NUMERIC_POST_CODE')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'GC_EVENT_NUMERIC_POST_POLICY')
    io.verify_digest(plan_path,binding['plan_sha256'],'GC_EVENT_NUMERIC_POST_PLAN')
    io.verify_digest(Path(plan['publication_terminal']['path']),plan['publication_terminal']['sha256'],
        'GC_EVENT_NUMERIC_POST_PUBLICATION')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):
        raise RuntimeError('GC_EVENT_NUMERIC_SOURCE_CHANGED')
    result = {'status':'COMPLETE_M1_TO_FIVE_TF_CLOCK_REFERENCE_ONLY','source_head':binding['source_head'],
        'plan_sha256':binding['plan_sha256'],'inputs':inputs,'source_publication':plan['publication_terminal'],
        'scan_start_utc':plan['scan_start_utc'],'scan_end_utc':plan['scan_end_utc'],'TEST_seal_utc':plan['TEST_seal_utc'],
        'm1_decoded_row_groups':m1_groups,'m1_rows':len(m1['bar_start_ns']),'timeframes':reports,'limits':proof,
        'reference_uses_published_float64_M1_sums':True,'raw_to_M1_formula_proven':False,
        'physical_network_latency_calibrated':False,'model_consumption_proven':False,
        'partial_scan_edge_buckets_excluded':True,'test_data_used':False,'model_forwards':0,'optimizer_updates':0}
    if event_book:
        result.update(status='COMPLETE_SCOPED_EVENT_BOOK_M1_TO_TF_REFERENCE',
            m1_book_decoded_row_groups=book_groups,strict_m5=strict_report,
            active_field_names=event_book_numeric_active_fields(),
            auxiliary_columns=sorted(GC_EVENT_BOOK_AUXILIARY_COLUMNS),
            native_availability_projection_proven=False,
            published_ofi_incident_masks_reused_without_raw_reconstruction=True)
    if trade:
        active=trade_numeric_active_fields()
        for tf,fields in active.items():
            for column in fields:
                if not reports[tf]['fields'][column]['numeric_reference_qualified']:
                    raise RuntimeError('GC_TRADE_REFERENCE_ACTIVE_FIELD_UNOBSERVED:'+tf+':'+column)
        result.update(status='COMPLETE_SCOPED_TRADE_CUMULATIVE_M1_TO_TF_REFERENCE',
            reference_uses_published_float64_M1_sums=False,reference_uses_published_uint64_M1_statistics=True,
            m1_trade_size_decoded_row_groups=trade_size_groups,all_m1_size_parity_rows=len(m1['bar_start_ns']),
            cumulative_m1_passes=1,active_field_names=active,auxiliary_columns=list(GC_TRADE_AUXILIARY_COLUMNS),
            unregistered_strict_outputs_are_auxiliary=True,native_availability_projection_proven=False,
            cumulative_reset_clock='UTC date, contract, invalid M1 or nonconsecutive minute',
            published_trade_masks_reused_without_raw_reconstruction=True)
    if smc:
        active,strict_active=smc_numeric_active_fields()
        for tf,fields in active.items():
            for column in fields:
                if not reports[tf]['fields'][column]['numeric_reference_qualified']:
                    raise RuntimeError('GC_SMC_REFERENCE_ACTIVE_FIELD_UNOBSERVED:'+tf+':'+column)
        if not strict_report['fields']['close_usd']['numeric_reference_qualified']:
            raise RuntimeError('GC_SMC_REFERENCE_BOOK_CLOSE_UNOBSERVED')
        result.update(status='COMPLETE_SCOPED_SMC_PRICE_M1_TO_TF_REFERENCE',
            reference_uses_published_float64_M1_sums=False,reference_uses_published_integer_M1_ohlc=True,
            active_field_names=active,strict_m5_active_field_names=strict_active,strict_m5_price=strict_report,
            integer_price_order_proof=price_order,unregistered_strict_outputs_are_auxiliary=True,
            native_availability_projection_proven=False,published_volume_masks_reused_without_raw_reconstruction=True,
            pivot_confirmation='three completed right candles; independent causal prefixes',
            observed_gap_semantics='pivot memory retained; zones and origins censored after whole missing same-contract candle')
    if profile:
        for tf,fields in profile_numeric_active_fields().items():
            measured=strict_report['fields'] if tf=='strict_m5' else reports[tf]['fields']
            for column in fields:
                if not measured[column]['numeric_reference_qualified']:
                    raise RuntimeError('GC_PROFILE_REFERENCE_ACTIVE_FIELD_UNOBSERVED:'+tf+':'+column)
        io.verify_digest(Path(plan['pending_field_inventory']['path']),plan['pending_field_inventory']['sha256'],
            'GC_PROFILE_REFERENCE_POST_PENDING_INVENTORY')
        result.update(status='COMPLETE_SCOPED_PROFILE_FOOTPRINT_M1_TO_TF_REFERENCE',
            reference_uses_published_float64_M1_sums=False,reference_uses_published_integer_M1_ladder=True,
            pending_field_inventory=plan['pending_field_inventory'],active_field_names=profile_numeric_active_fields(),
            ladder_decoded_row_groups=ladder_groups,ladder_reference=profile_report,strict_m5_profile=strict_report,
            unregistered_strict_outputs_are_auxiliary=True,native_availability_projection_proven=False,
            profile_projection='exact M5 available equals TF available; no older-valid backsearch',
            dynamic_profile_clock='consecutive closed M5; NY18 IANA session anchor',
            raw_to_M1_formula_proven=False,published_volume_signed_masks_reused_without_raw_reconstruction=True)
    if basis:
        for scope,fields in basis_numeric_active_fields().items():
            measured=(reports[scope.removeprefix('strict_')]['strict_fields'] if scope.startswith('strict_') else reports[scope]['fields'])
            for column in fields:
                if not measured[column]['numeric_reference_qualified']:
                    raise RuntimeError('GC_BASIS_REFERENCE_ACTIVE_FIELD_UNOBSERVED:'+scope+':'+column)
        for name in ('source_basis_plan','publication_result','pending_field_inventory'):
            item=plan[name];io.verify_digest(Path(item['path']),item['sha256'],'GC_BASIS_REFERENCE_POST_METADATA')
        io.verify_digest(Path(inputs['spot_manifest']['path']),inputs['spot_manifest']['sha256'],'GC_BASIS_REFERENCE_POST_MANIFEST')
        result.update(status='COMPLETE_SCOPED_CLOSED_M1_SPOT_FUTURES_BASIS_REFERENCE',
            reference_uses_published_float64_M1_sums=False,reference_uses_published_binary64_spot_quotes=True,
            active_field_names=basis_numeric_active_fields(),source_basis_plan=plan['source_basis_plan'],
            publication_result=plan['publication_result'],pending_field_inventory=plan['pending_field_inventory'],
            spot_decoded_row_groups=spot_groups,spot_reference=spot_report,
            inactive_strict_D1_is_auxiliary=True,native_availability_projection_proven=False,
            source_clock_semantics='same closed M1 interval; no event-level simultaneity',
            age_semantics='TFavailable minus observed M1available; native decision transform excluded',
            source_final_M1_excluded_when_available_equals_TEST_seal=True)
    output = staging/(prefix+'_RESULT.json'); io.write_json(output,result)
    io.write_json(staging/(prefix+'_TERMINAL.json'), {'status':'COMPLETE','exit_code':0,
        'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'result_sha256':io.sha256_file(output)})
    publish_bundle_directory_noreplace(staging,root/(prefix+'_RESULTS'))
    print(prefix+'_REFERENCE_COMPLETE',flush=True)



GC_NATIVE_PROJECTION_PLAN_ID='NATIVE_M5_GC_NATIVE_PROJECTION_REFERENCE_20261021_126'
GC_NATIVE_PROJECTION_POLICY_KEY='gc_native_projection_numeric_reference_126'
GC_NATIVE_AGE_COLUMNS=frozenset(('observed_profile_snapshot_age_minutes','observed_spot_basis_age_minutes'))

def _projection_integer_clock(values, label, *, grid=None):
    a=np.asarray(values)
    if (a.ndim!=1 or a.dtype!=np.dtype('int64') or (len(a)>1 and np.any(np.diff(a)<=0))
            or (grid is not None and np.any(a%grid))):
        raise RuntimeError(label)
    return a

def gc_native_decision_stamps(source_barstarts, *, seal=None):
    """M5 surface stores BARSTART; GC observes this bar only after its close."""
    a=_projection_integer_clock(source_barstarts,'GC_NATIVE_PROJECTION_CLOCK',grid=5*MINUTE)
    if len(a) and a[-1]>np.iinfo(np.int64).max-5*MINUTE:raise RuntimeError('GC_NATIVE_PROJECTION_CLOCK')
    decisions=a+5*MINUTE
    if seal is not None and len(decisions) and decisions[-1]>=seal:raise RuntimeError('GC_NATIVE_PROJECTION_CLOCK')
    return decisions

def reference_consumed_surface_rows(source_barstarts, split_barstarts, *, sequence_length):
    """Exact emitted row p consumes [p-L+1,p]; interval union needs no seq decode."""
    source=_projection_integer_clock(source_barstarts,'GC_NATIVE_PROJECTION_ROW_UNION_CLOCK',grid=5*MINUTE)
    if not len(source) or not isinstance(sequence_length,int) or sequence_length<1:
        raise RuntimeError('GC_NATIVE_PROJECTION_ROW_UNION_HISTORY')
    masks={};counts={};split_rows={}
    for split,values in split_barstarts.items():
        clocks=_projection_integer_clock(values,'GC_NATIVE_PROJECTION_ROW_UNION_CLOCK',grid=5*MINUTE)
        positions=np.searchsorted(source,clocks)
        if np.any(positions>=len(source)) or not np.array_equal(source[positions],clocks):
            raise RuntimeError('GC_NATIVE_PROJECTION_ROW_UNION_CLOCK')
        if np.any(positions<sequence_length-1):raise RuntimeError('GC_NATIVE_PROJECTION_ROW_UNION_HISTORY')
        delta=np.zeros(len(source)+1,np.int64)
        np.add.at(delta,positions-sequence_length+1,1);np.add.at(delta,positions+1,-1)
        masks[split]=np.cumsum(delta[:-1])>0;counts[split]=int(masks[split].sum());split_rows[split]=len(clocks)
    masks['union']=np.logical_or.reduce(list(masks.values())) if masks else np.zeros(len(source),bool)
    selected=np.flatnonzero(masks['union'])
    return masks,{'source_rows':len(source),'sequence_length':sequence_length,'split_rows':split_rows,
        'consumed_rows_by_split':counts,'union_rows':len(selected),
        'first_union_row':int(selected[0]) if len(selected) else None,'last_union_row':int(selected[-1]) if len(selected) else None,
        'first_decision_ns':int(source[selected[0]]+5*MINUTE) if len(selected) else None,
        'last_decision_ns':int(source[selected[-1]]+5*MINUTE) if len(selected) else None}

def native_projection_layout(fields, ctx_names, signal_names, surface_names):
    names=[name for item in fields for name in (item['name'],item['availability_field'])]
    if (len(set(names))!=len(names) or len(set(ctx_names))!=len(ctx_names)
            or len(set(signal_names))!=len(signal_names) or list(signal_names)!=list(surface_names)):
        raise RuntimeError('GC_NATIVE_PROJECTION_FIELD_LAYOUT')
    ctx={name:i for i,name in enumerate(ctx_names)};signal={name:i for i,name in enumerate(signal_names)}
    if any(name not in ctx or 'ctx_cont.'+name not in signal for name in names):
        raise RuntimeError('GC_NATIVE_PROJECTION_FIELD_LAYOUT')
    return {'names':names,'ctx_indices':np.array([ctx[n] for n in names],np.int64),
        'signal_indices':np.array([signal['ctx_cont.'+n] for n in names],np.int64)}

def reference_gc_native_projection(fields, tables, decisions, *, cursors=None):
    """Independent two-pointer asof, own domains, then observed-only binary32 cast.

    Cursors is opaque mutable state keyed by table-object identity, for strictly
    advancing batches. Only selected rows are inspected for field-domain errors;
    a poisoned future bucket cannot affect an earlier decision.
    """
    query=_projection_integer_clock(decisions,'GC_NATIVE_PROJECTION_CLOCK')
    state={} if cursors is None else cursors;joins={};prepared_clocks={};output=np.zeros((len(query),2*len(fields)),np.float32);reports={}
    for item in fields:
        table=tables[item['table']];key=id(table)
        if key in joins:continue
        old=state.get(key,{'row':-1,'decision':None})
        clock=(_projection_integer_clock(table['available_ns'],'GC_NATIVE_PROJECTION_SOURCE_CLOCK')
            if key not in state else np.asarray(table['available_ns']))
        if len(query) and old['decision'] is not None and query[0]<=old['decision']:
            raise RuntimeError('GC_NATIVE_PROJECTION_CLOCK')
        take=np.empty(len(query),np.int64);row=int(old['row'])
        for i,decision in enumerate(query):
            while row+1<len(clock) and int(clock[row+1])<=int(decision):row+=1
            take[i]=row
        joins[key]=(clock,take)
        if len(query):state[key]={'row':row,'decision':int(query[-1])}
    for j,item in enumerate(fields):
        table=tables[item['table']];clock,index=joins[id(table)];selected=index>=0
        values=np.full(len(query),np.nan);source=np.zeros(len(query),np.int64);defined=np.zeros(len(query),bool)
        if len(clock):
            take=np.maximum(index,0);raw=np.asarray(table[item['column']])
            if raw.ndim!=1 or len(raw)!=len(clock):raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
            values[selected]=raw[index[selected]].astype(np.float64)
            observed=selected&np.isfinite(values)
            if item.get('validity_column'):
                valid=np.asarray(table[item['validity_column']])
                if valid.ndim!=1 or len(valid)!=len(clock):raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
                admitted=valid[index[selected]].astype(np.float64)
                if not np.isin(admitted[np.isfinite(admitted)],(0,1)).all():raise RuntimeError('GC_NATIVE_PROJECTION_VALIDITY_DOMAIN')
                own=np.zeros(len(query),bool);own[selected]=np.isfinite(admitted)&(admitted==1);observed&=own
            clock_column=item.get('source_clock_column') or 'available_ns';clock_key=(id(table),clock_column)
            if clock_key not in prepared_clocks:
                raw_clock=np.asarray(table[clock_column])
                if raw_clock.ndim!=1 or len(raw_clock)!=len(clock):raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
                chosen=raw_clock[index[selected]];finite=np.isfinite(chosen);finite_values=chosen[finite]
                # Range precedes conversion; float64 2**63 cannot wrap to int64.
                if (np.any(finite_values<-(2**63)) or np.any(finite_values>=2**63)
                        or (not np.issubdtype(finite_values.dtype,np.integer) and np.any(np.trunc(finite_values)!=finite_values))):
                    raise RuntimeError('GC_NATIVE_PROJECTION_SOURCE_CLOCK_PRECISION')
                selected_rows=np.flatnonzero(selected);defined[selected_rows[finite]]=True
                source[selected_rows[finite]]=finite_values.astype(np.int64)
                if np.any(defined&((source>query)|(source>clock[take]))):
                    raise RuntimeError('GC_NATIVE_PROJECTION_FUTURE_SOURCE_CLOCK:'+item['name'])
                prepared_clocks[clock_key]=(source,defined)
            source,defined=prepared_clocks[clock_key];observed&=defined
        else:observed=np.zeros(len(query),bool)
        transform=item.get('transform')
        if transform is None and item['column'] in GC_NATIVE_AGE_COLUMNS:transform='decision_age_minutes'
        if transform:
            if transform!='decision_age_minutes' or item['column'] not in GC_NATIVE_AGE_COLUMNS:
                raise RuntimeError('GC_NATIVE_PROJECTION_TRANSFORM')
            values=(query-source)/float(MINUTE)
        if item.get('role')=='structural_observation_mask' and not np.isin(values[observed],(0,1)).all():
            raise RuntimeError('GC_NATIVE_PROJECTION_STRUCTURAL_VALUE_DOMAIN')
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            output[observed,2*j]=values[observed].astype(np.float32)
        if not np.isfinite(output[:,2*j]).all():raise RuntimeError('GC_NATIVE_PROJECTION_FLOAT32_OVERFLOW')
        output[:,2*j+1]=observed.astype(np.float32)
        reports[item['name']]={'observed':int(observed.sum()),'missing':int((~observed).sum()),
            'observed_zero':int((observed&(output[:,2*j]==0)).sum()),'nonzero':int((observed&(output[:,2*j]!=0)).sum()),
            'age_transformed':bool(transform),'positive_age':int((observed&(values>0)).sum()) if transform else 0,
            'source_clock_before_bucket':int((defined&selected&(source<clock[np.maximum(index,0)])).sum()) if len(clock) else 0,
            'bit_errors':0}
    return output,reports

def projection_numeric_footer(path, seal, *, kind):
    """Metadata-only barrier; GC publication is sealed by bucket BARSTART.

    Its final partial anchored H4/D1 bucket starts before TEST but has later
    availability. Keep that published row; the driver projects only availability
    <= the last consumed development decision. Availability bounds below remain
    the row-group selection bounds, never a replacement bar-start filter.
    """
    file=pq.ParquetFile(path);schema=file.schema_arrow;clock='available_ns' if kind=='gc' else 'time'
    if kind not in ('surface','train','val','gc') or clock not in schema.names:
        raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
    dtype=schema.field(clock).type
    if (kind=='gc' and (dtype!=pa.int64() or 'bar_start_ns' not in schema.names
            or schema.field('bar_start_ns').type!=pa.int64())) or (kind!='gc' and dtype!=pa.timestamp('ns',tz='UTC')):
        raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
    if kind=='surface':
        if 'ctx_cont' not in schema.names:raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
        nested=schema.field('ctx_cont').type
        if not pa.types.is_fixed_size_list(nested) or nested.value_type!=pa.float32() or nested.list_size!=2542:
            raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
    groups=[];sizes=[];bounds=[]
    for i in range(file.metadata.num_row_groups):
        row=file.metadata.row_group(i)
        matches=[j for j in range(row.num_columns) if row.column(j).path_in_schema==clock]
        if len(matches)!=1:raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
        stat=row.column(matches[0]).statistics
        if stat is None or not stat.has_min_max or stat.null_count!=0:raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
        lo,hi=(int(stat.min),int(stat.max)) if kind=='gc' else (pd.Timestamp(stat.min).value,pd.Timestamp(stat.max).value)
        if lo>hi:raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
        if kind=='gc':
            starts=[j for j in range(row.num_columns) if row.column(j).path_in_schema=='bar_start_ns']
            if len(starts)!=1:raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
            start_stat=row.column(starts[0]).statistics
            if start_stat is None or not start_stat.has_min_max or start_stat.null_count!=0:
                raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
            if int(start_stat.min)>int(start_stat.max):raise RuntimeError('GC_NATIVE_PROJECTION_SCHEMA')
            if int(start_stat.max)>=seal:raise RuntimeError('GC_NATIVE_PROJECTION_TEST_ROW_GROUP')
        elif hi>=seal:raise RuntimeError('GC_NATIVE_PROJECTION_TEST_ROW_GROUP')
        groups.append(i);sizes.append(row.num_rows);bounds.append([lo,hi])
    return {'groups':groups,'rows':file.metadata.num_rows,'row_group_rows':sizes,'clock_bounds':bounds}


def run_gc_native_projection_numeric(binding_path):
    """One scoped publication-to-native proof; never reads seq/snap/signal."""
    prefix='GC_NATIVE_PROJECTION';root=Path('/home/andre2/GX1_RUNS')/GC_NATIVE_PROJECTION_PLAN_ID
    if binding_path!=root/(prefix+'_EXECUTION_BINDING.json'):raise RuntimeError(prefix+'_BINDING_PATH')
    io.require_regular_file(binding_path,prefix+'_BINDING');binding=io.read_json(binding_path);plan_path=root/'PLAN.json'
    io.verify_digest(plan_path,binding['plan_sha256'],prefix+'_PLAN');plan=io.read_json(plan_path)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],prefix+'_POLICY')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work'][GC_NATIVE_PROJECTION_POLICY_KEY]
    if (plan['schema_version']!='gx1_gc_native_projection_audit_plan_v1' or plan['plan_id']!=GC_NATIVE_PROJECTION_PLAN_ID
            or plan['run_root']!=str(root) or plan['policy_key']!=GC_NATIVE_PROJECTION_POLICY_KEY
            or plan['max_invocations']!=1 or plan['batch_rows']!=256 or gate['status']!='EXECUTION_CLAIMED'
            or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0
            or gate['execution_authorized'] is not False or gate['relaunch_allowed'] is not False
            or gate['plan']!={'path':str(plan_path),'sha256':binding['plan_sha256']}
            or policy['training_enabled'] is not False or io.git_head()!=binding['source_head']
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):
        raise RuntimeError(prefix+'_AUTHORITY')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError(prefix+'_CODE_INVENTORY')
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,prefix+'_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    io.write_json(root/(prefix+'_CLAIM.json'),{'invocation':1,'plan_sha256':binding['plan_sha256'],'source_head':binding['source_head']})
    staging=root/('.'+prefix+'_RESULTS.staging');staging.mkdir(exist_ok=False)
    seal_stamp=pd.Timestamp(plan['TEST_seal_utc'])
    from gx1.contracts.gc_source_disposition_v1 import WINDOW_END_NS
    if seal_stamp.tzinfo is None or seal_stamp.value!=WINDOW_END_NS:raise RuntimeError(prefix+'_SEAL')
    seal=seal_stamp.value;small={};identities={};hash_files={}
    identity=lambda st:(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def metadata(ref):
        path=Path(ref['path']);digest=ref['sha256']
        if str(path) in small:
            if small[str(path)]!=digest:raise RuntimeError(prefix+'_METADATA_COLLISION')
            return io.read_json(path)
        io.require_regular_file(path,prefix+'_METADATA')
        if not path.is_absolute() or path!=path.resolve():raise RuntimeError(prefix+'_CANONICAL_PATH')
        before=identity(path.stat());io.verify_digest(path,digest,prefix+'_METADATA');value=io.read_json(path)
        if identity(path.stat())!=before:raise RuntimeError(prefix+'_METADATA_CHANGED')
        small[str(path)]=digest;identities[str(path)]=before;return value
    design=metadata(plan['design']);addendum=metadata(plan['design_signal_alias_addendum'])
    if (design['run_id']!=GC_NATIVE_PROJECTION_PLAN_ID or design['references']!=plan['references']
            or addendum['design']!=plan['design'] or addendum['context_signal_alias_proof']!=plan['context_signal_alias_proof']):
        raise RuntimeError(prefix+'_DESIGN')
    refs=plan['references'];registry=metadata(refs['registry']);index=metadata(refs['published_m1_tf_numeric_index'])
    publication=metadata(refs['publication_terminal']);surfaces=metadata(refs['surface_bindings'])
    reconstruction_bindings=metadata(refs['source_reconstruction_bindings']);normalization=metadata(refs['normalization'])
    reconstruction={split:metadata(ref) for split,ref in refs['reconstruction'].items()}
    fields=registry['fields'];expected=design['expected_contract'];field_names={item['name'] for item in fields}
    if (len(fields)!=1205 or len(field_names)!=1205 or index['registry']!=refs['registry']
            or index['status']!='COMPLETE_SCOPED_ALL_1205_REGISTRY_VALUE_FIELDS'
            or index['qualified_active_registry_fields']!=1205 or set(index['fields'])!=field_names
            or index['missing_fields'] or index['foreign_fields'] or index['duplicate_fields']
            or index['test_data_used'] is not False or publication['status']!='COMPLETE' or publication['exit_code']!=0
            or publication['source_unchanged'] is not True or publication['test_data_used'] is not False
            or surfaces['active_registry']!=refs['registry'] or surfaces['test_data_used'] is not False
            or reconstruction_bindings['decision']!='PASS_CURRENT_PHYSICAL_METADATA_AND_FRESH_SOURCE_RECONSTRUCTION'
            or reconstruction_bindings['test_data_used'] is not False or normalization['decision']!='PASS'
            or normalization['test_accessed'] is not False or normalization['test_fit_rows']!=0 or normalization['val_fit_rows']!=0):
        raise RuntimeError(prefix+'_REUSED_PROOF')
    surface=surfaces['m5_feature_surface'];surface_manifest=metadata({'path':surface['manifest_path'],'sha256':surface['manifest_sha256']})
    manifests={};physical={surface['path']:{'sha256':surface['sha256'],'kind':'surface'}}
    for split in ('train','val'):
        report=reconstruction[split];bound=reconstruction_bindings['reports'][split]
        if (bound['path']!=refs['reconstruction'][split]['path'] or bound['sha256']!=refs['reconstruction'][split]['sha256']
                or bound['rows']!=report['rows'] or bound['sequence_source_chain_sha256']!=report['sequence_source_chain_sha256']
                or report['decision']!='PASS' or len(report['checks'])!=8 or not all(v is True for v in report['checks'].values())
                or report['rows']!=expected[split+'_rows'] or report['sequence_shape']!=[report['rows'],96,2725]
                or report['snapshot_shape']!=[report['rows'],2725] or report['feature_surface_rows']!=expected['surface_rows']
                or any(report['feature_surface_'+key]!=surface[target] for key,target in
                    (('path','path'),('sha256','sha256'),('manifest_path','manifest_path'),('manifest_sha256','manifest_sha256')))):
            raise RuntimeError(prefix+'_SOURCE_RECONSTRUCTION')
        manifests[split]=metadata({'path':report['manifest_path'],'sha256':report['manifest_sha256']})
        physical[report['parquet_path']]={'sha256':report['parquet_sha256'],'kind':split}
    norm_lineage=normalization['contract']['lineage']
    if (norm_lineage['train_parquet_path']!=reconstruction['train']['parquet_path']
            or norm_lineage['train_parquet_sha256']!=reconstruction['train']['parquet_sha256']
            or norm_lineage['train_manifest_sha256']!=reconstruction['train']['manifest_sha256']
            or norm_lineage['local_fit_row_count']!=118800 or norm_lineage['test_fit_row_count']!=0
            or norm_lineage['val_fit_row_count']!=0):raise RuntimeError(prefix+'_NORMALIZATION_REUSE')
    for item in fields:
        evidence=index['fields'][item['name']]
        if (evidence['registry_table']!=item['table'] or evidence['registry_column']!=item['column']
                or evidence['availability_field']!=item['availability_field']
                or evidence['statistics']['numeric_reference_qualified'] is not True
                or evidence['statistics']['finite_published_values_compared']<=0):
            raise RuntimeError(prefix+'_NUMERIC_INDEX_FIELD')
    columns={};logical={};publication_root=Path(refs['publication_terminal']['path']).parent
    for key,ref in registry['tables'].items():
        path=Path(ref['path']);artifact=publication['artifacts'][path.name]
        if path!=publication_root/path.name or artifact['sha256']!=ref['sha256']:
            raise RuntimeError(prefix+'_PUBLISHED_TABLE_BINDING')
        previous=physical.get(str(path))
        if previous and previous['sha256']!=ref['sha256']:raise RuntimeError(prefix+'_PHYSICAL_ALIAS_COLLISION')
        physical[str(path)]={'sha256':ref['sha256'],'kind':'gc','bytes':artifact['bytes']};logical[key]=str(path)
        columns.setdefault(str(path),{'available_ns'})
    for item in fields:
        needed=columns[logical[item['table']]];needed.add(item['column'])
        needed.update(item[name] for name in ('validity_column','source_clock_column') if item.get(name))
    if len(physical)!=13 or len(columns)!=10 or len(logical)!=14:raise RuntimeError(prefix+'_PHYSICAL_INVENTORY')
    # One whole-file digest per unique physical path. Later guards use stat only.
    for name,item in physical.items():
        path=Path(name);io.require_regular_file(path,prefix+'_INPUT')
        if not path.is_absolute() or path!=path.resolve():raise RuntimeError(prefix+'_CANONICAL_PATH')
        before=identity(path.stat());io.verify_digest(path,item['sha256'],prefix+'_INPUT');after=identity(path.stat())
        if before!=after or ('bytes' in item and after[3]!=item['bytes']):raise RuntimeError(prefix+'_INPUT_CHANGED')
        identities[name]=after;hash_files[name]={'sha256':item['sha256'],'bytes':after[3],'whole_file_hash_passes':1}
    # Every source/split/GC footer is admitted before any clock or value payload.
    footers={name:projection_numeric_footer(Path(name),seal,kind=item['kind']) for name,item in physical.items()}
    for name,needed in columns.items():
        schema=pq.ParquetFile(name).schema_arrow
        if any(column not in schema.names or not (pa.types.is_boolean(schema.field(column).type)
                or pa.types.is_integer(schema.field(column).type) or pa.types.is_floating(schema.field(column).type)) for column in needed):
            raise RuntimeError(prefix+'_GC_COLUMN_SCHEMA')
    def read_clock(path):
        table=pq.ParquetFile(path).read_row_groups(footers[path]['groups'],columns=['time'],use_threads=False)
        if table['time'].null_count:raise RuntimeError(prefix+'_CLOCK')
        return table['time'].cast(pa.int64()).to_numpy()
    source_clock=read_clock(surface['path']);split_clocks={split:read_clock(report['parquet_path']) for split,report in reconstruction.items()}
    if len(source_clock)!=expected['surface_rows'] or any(len(split_clocks[s])!=expected[s+'_rows'] for s in ('train','val')):
        raise RuntimeError(prefix+'_ROW_COUNTS')
    for split,clock in split_clocks.items():
        bounds=manifests[split]['ts_min_max_by_split'][split]
        if int(clock[0])!=pd.Timestamp(bounds['ts_min']).value or int(clock[-1])!=pd.Timestamp(bounds['ts_max']).value:
            raise RuntimeError(prefix+'_SPLIT_CLOCK_BINDING')
    masks,row_union=reference_consumed_surface_rows(source_clock,split_clocks,sequence_length=96)
    selected=np.flatnonzero(masks['union']);decisions=gc_native_decision_stamps(source_clock[selected],seal=seal)
    if not len(selected):raise RuntimeError(prefix+'_EMPTY_ROW_UNION')
    contract=manifests['train']['feature_contract'];ctx_names=contract['ctx_cont_names'];signal_names=contract['signal_bridge_fields']
    if manifests['val']['feature_contract']!=contract:raise RuntimeError(prefix+'_SPLIT_FIELD_CONTRACT')
    layout=native_projection_layout(fields,ctx_names,signal_names,surface_manifest['feature_field_order'])
    if (len(ctx_names)!=2542 or len(signal_names)!=2725 or surface_manifest['rows']!=len(source_clock)
            or not np.array_equal(layout['ctx_indices'],np.arange(132,2542))
            or not np.array_equal(layout['signal_indices'],np.arange(315,2725))):raise RuntimeError(prefix+'_ACTUAL_2410_LANES')
    alias=metadata(plan['context_signal_alias_proof']);entry=alias['entry'];population_start=entry['excluded_pre_history_row_count']
    alias_input=alias['input_bindings']['m5_feature_surface'];hashes=entry['local_field_hashes']
    if (alias['schema_version']!='entry_cross_surface_input_overlap_v5' or alias['decision']!='PASS' or alias['failures']
            or any(alias_input[key]!=surface[key] for key in ('path','sha256','manifest_path','manifest_sha256'))
            or alias_input['rows']!=len(source_clock) or alias_input['pair_generation_id']!=surface_manifest['pair_generation_id']
            or entry['source_row_count']!=len(source_clock) or population_start+entry['row_count']!=len(source_clock)
            or not 0<=population_start<len(source_clock) or selected[0]<population_start
            or int(source_clock[population_start])!=entry['first_time_ns'] or int(source_clock[-1])!=entry['last_time_ns']):
        raise RuntimeError(prefix+'_SIGNAL_ALIAS_POPULATION')
    import re
    for name in layout['names']:
        digest=hashes.get('local.ctx_cont.'+name)
        if not isinstance(digest,str) or re.fullmatch('[0-9a-f]{64}',digest) is None or hashes.get('local.signal.ctx_cont.'+name)!=digest:
            raise RuntimeError(prefix+'_SIGNAL_ALIAS_HASH:'+name)
    tables={};table_reads={}
    for path,needed in columns.items():
        footer=footers[path];groups=[i for i,(lo,hi) in enumerate(footer['clock_bounds']) if lo<=int(decisions[-1])]
        table=pq.ParquetFile(path).read_row_groups(groups,columns=sorted(needed),use_threads=False)
        values={name:table[name].to_numpy() for name in needed};clock=values['available_ns']
        _projection_integer_clock(clock,prefix+'_SOURCE_CLOCK');keep=clock<=decisions[-1]
        values={name:np.asarray(value)[keep] for name,value in values.items()}
        for key,target in logical.items():
            if target==path:tables[key]=values
        table_reads[path]={'logical_tables':[key for key,target in logical.items() if target==path],
            'projected_columns':sorted(needed),'decoded_rows':len(clock),'retained_rows':int(keep.sum()),'decoded_row_groups':len(groups)}
    counters={split:{key:np.zeros(1205,np.int64) for key in ('observed','missing','observed_zero','nonzero','positive_age','bit_errors')}
        for split in ('train','val','union')}
    ages=np.array([item.get('transform')=='decision_age_minutes' or item['column'] in GC_NATIVE_AGE_COLUMNS for item in fields])
    cursor={};visited=np.zeros(len(source_clock),bool);ctx_rows=0;ctx_batches=0;compared=0;row_offset=0
    source_file=pq.ParquetFile(surface['path'])
    for group,size in enumerate(footers[surface['path']]['row_group_rows']):
        if not masks['union'][row_offset:row_offset+size].any():row_offset+=size;continue
        batch_offset=row_offset
        for batch in source_file.iter_batches(batch_size=256,row_groups=[group],columns=['time','ctx_cont'],use_threads=False):
            n=batch.num_rows;ctx_rows+=n;ctx_batches+=1;positions=np.arange(batch_offset,batch_offset+n);take=masks['union'][positions]
            if take.any():
                stamps=batch.column(batch.schema.get_field_index('time')).cast(pa.int64()).to_numpy()
                if not np.array_equal(stamps,source_clock[positions]):raise RuntimeError(prefix+'_STREAM_CLOCK')
                nested=batch.column(batch.schema.get_field_index('ctx_cont'))
                if nested.null_count or nested.values.null_count:raise RuntimeError(prefix+'_STREAM_NULL')
                matrix=nested.flatten().to_numpy().reshape(n,2542)
                actual=np.ascontiguousarray(matrix[take][:,layout['ctx_indices']],dtype=np.float32)
                queries=gc_native_decision_stamps(stamps[take],seal=seal)
                reference,_=reference_gc_native_projection(fields,tables,queries,cursors=cursor)
                if not np.isfinite(actual).all() or not np.array_equal(actual.view(np.uint32),reference.view(np.uint32)):
                    mismatch=np.argwhere(actual.view(np.uint32)!=reference.view(np.uint32))
                    first=mismatch[0] if len(mismatch) else (0,0)
                    raise RuntimeError(prefix+'_EXACT_BITS:'+str(int(queries[first[0]]))+':'+layout['names'][first[1]])
                chosen=positions[take];visited[chosen]=True;compared+=len(chosen)
                obs=reference[:,1::2]==1;value=reference[:,::2]
                for split in ('train','val','union'):
                    admitted=masks[split][chosen];own=obs[admitted];v=value[admitted];count=counters[split]
                    count['observed']+=own.sum(axis=0);count['missing']+=(~own).sum(axis=0)
                    count['observed_zero']+=(own&(v==0)).sum(axis=0);count['nonzero']+=(own&(v!=0)).sum(axis=0)
                    count['positive_age']+=(own&(v>0)&ages).sum(axis=0)
            batch_offset+=n
        if batch_offset!=row_offset+size:raise RuntimeError(prefix+'_STREAM_ROW_COUNT')
        row_offset+=size
    if not np.array_equal(visited,masks['union']) or compared!=len(selected):raise RuntimeError(prefix+'_CONSUMED_ROW_COVERAGE')
    reports={}
    for split,count in counters.items():
        reports[split]={item['name']:{key:int(values[j]) for key,values in count.items()} for j,item in enumerate(fields)}
        for j,item in enumerate(fields):
            reports[split][item['name']]['age_transformed']=bool(ages[j])
            reports[split][item['name']]['numeric_projection_qualified']=bool(count['observed'][j]>0)
            if split=='union' and count['observed'][j]==0:raise RuntimeError(prefix+'_ACTIVE_FIELD_UNOBSERVED:'+item['name'])
    for path,before in identities.items():
        if identity(Path(path).stat())!=before:raise RuntimeError(prefix+'_POST_INPUT_CHANGED')
    for path,digest in small.items():io.verify_digest(Path(path),digest,prefix+'_POST_METADATA')
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,prefix+'_POST_CODE')
    io.verify_digest(plan_path,binding['plan_sha256'],prefix+'_POST_PLAN')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],prefix+'_POST_POLICY')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):
        raise RuntimeError(prefix+'_POST_SOURCE_CHANGED')
    result={'schema_version':'gx1_gc_native_projection_numeric_result_v1','status':'COMPLETE_SCOPED_ALL_1205_GC_NATIVE_PROJECTION',
        'plan_sha256':binding['plan_sha256'],'source_head':binding['source_head'],'source_unchanged':True,'limits':proof,
        'registry':refs['registry'],'reused_published_numeric_index':refs['published_m1_tf_numeric_index'],
        'reused_source_reconstruction':refs['reconstruction'],'reused_normalization_metadata':refs['normalization'],
        'context_signal_alias_proof':plan['context_signal_alias_proof'],'source_row_union':row_union,
        'value_fields':1205,'availability_fields':1205,'context_lane_range':[132,2542],'signal_lane_range':[315,2725],
        'exact_float32_bit_errors':0,'decision_clock':'M5surface bar_start_ns + 300 seconds',
        'independent_two_pointer_latest_bucket':True,'no_older_valid_backsearch':True,'float32_cast_after_own_validity':True,
        'field_reports':reports,'physical_inputs':hash_files,'deduplicated_physical_file_count':len(physical),
        'whole_file_hash_passes':len(hash_files),'bound_physical_file_bytes':sum(item['bytes'] for item in hash_files.values()),
        'gc_table_reads':table_reads,'source_clock_decoded_rows':len(source_clock),'split_clock_decoded_rows':{s:len(c) for s,c in split_clocks.items()},
        'ctx_cont_projected_columns':['time','ctx_cont'],'ctx_cont_decoded_rows':ctx_rows,'ctx_cont_decoded_batches':ctx_batches,
        'selected_source_rows_compared':compared,'batch_rows':256,'disk_io_bytes_measured':False,
        'native_sequence_or_signal_payload_decoded':False,'model_forwards':0,'optimizer_updates':0,'test_data_used':False,
        'normalization_fit_performed':False,'individual_learned_use_proven':False,'raw_to_M1_formula_proven':False,
        'other_technical_numeric_formulas_proven':False,'learning_qualified':False,'economic_edge_qualified':False}
    output=staging/(prefix+'_RESULT.json');io.write_json(output,result)
    io.write_json(staging/(prefix+'_TERMINAL.json'),{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
        'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(output)})
    publish_bundle_directory_noreplace(staging,root/(prefix+'_RESULTS'));print(prefix+'_REFERENCE_COMPLETE',flush=True)


TECHNICAL_NUMERIC_PLAN_ID='NATIVE_M5_TECHNICAL_NUMERIC_REFERENCE_20261021_127'
TECHNICAL_NUMERIC_POLICY_KEY='technical_numeric_reference_127'
TECHNICAL_NUMERIC_FIELDS=('atr_bps_14','rsi14_centered','mom_5_atr','mom_20_atr','close_open_atr',
    'ema20_dist_atr','ema50_dist_atr','ema100_dist_atr','ema200_dist_atr',
    'ema20_slope_atr','ema50_slope_atr','ema100_slope_atr','ema200_slope_atr','ema_stack_aligned_v2',
    'vwap_rolling5_dist_atr','vwap20_dist_atr','vwap96_dist_atr','vwap_rolling5_slope_atr',
    'bb_position','bb_width_atr','adx14','di_spread_signed','trend_state_age_bars')

def _technical_up(value):
    import math
    if isinstance(value,(float,int,np.floating,np.integer)):
        out=math.nextafter(float(value),math.inf)
        if not math.isfinite(out):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
        return out
    out=np.nextafter(value,np.inf)
    if not np.isfinite(out).all():raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
    return out

def _technical_uadd(*values):
    # Each named nonnegative radius node rounds outward; not one bump after
    # a compound expression. These helpers do not evaluate value intervals.
    out=0.
    for value in values:out=_technical_up(out+value)
    return out

def _technical_umul(a,b):return _technical_up(a*b)

def _technical_udiv(a,b):return _technical_up(a/b)

def _technical_rho(value):
    # ERROR_SPEC: IEEE64 RN residual, with both nodes outward and eta64.
    if isinstance(value,(float,int,np.floating,np.integer)):
        a=abs(float(value));u=2.**-53
        return _technical_udiv(_technical_umul(u,a),1.-u)
    a=np.abs(value);u=2.**-53
    return _technical_udiv(_technical_umul(u,a),1.-u)

def _technical_vector(values):
    a=np.asarray(values)
    if a.dtype!=np.dtype('float64') or a.ndim!=1 or not np.isfinite(a).all():
        raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    return a

def _technical_seed(values, *, divide=True):
    from fractions import Fraction
    import math
    n=len(values);exact=sum((Fraction.from_float(float(x)) for x in values),Fraction())
    exact=exact/n if divide else exact;center=float(exact)
    oracle_error=float(abs(Fraction.from_float(center)-exact))
    if not oracle_error and all(float(x)==float(values[0]) for x in values):
        numerator,denominator=float(values[0]).as_integer_ratio()
        # Every possible partial sum is then exact in binary64, not just the
        # chosen independent seed. This preserves genuine constant EMA ties.
        if abs(numerator).bit_length()+(n-1).bit_length()<=53:return center,0.
    u=2.**-53;gamma=(n-1)*u/(1-(n-1)*u)
    total=math.fsum(abs(float(x)) for x in values)
    error=_technical_umul(_technical_up(gamma),_technical_up(total))
    if divide:error=_technical_uadd(_technical_udiv(error,n),_technical_rho(center))
    return center,float(_technical_uadd(error,_technical_up(oracle_error)))

def reference_technical_ema(close, span):
    x=_technical_vector(close)
    if isinstance(span,bool) or not isinstance(span,int) or span<1:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    out=np.full(len(x),np.nan);bound=out.copy()
    if len(x)<span:return out,bound
    out[span-1],bound[span-1]=_technical_seed(x[:span]);a=2./(span+1.)
    from fractions import Fraction
    alpha_error=float(abs(Fraction.from_float(a)-Fraction(2,span+1)))
    for i in range(span,len(x)):
        m=float(out[i-1]);e=float(bound[i-1]);d=float(x[i])-m;p=a*d;y=m+p
        out[i]=y
        if e==0. and d==0.:bound[i]=0.;continue
        # The shared alpha is the exact same represented declared constant.
        # Compare both RN evaluations to the affine map, preserving contraction.
        rd=_technical_uadd(_technical_rho(_technical_uadd(abs(d),e)),_technical_rho(d))
        rp=_technical_uadd(_technical_rho(_technical_umul(a,_technical_uadd(abs(d),e,rd))),_technical_rho(p))
        affine=_technical_uadd(_technical_umul(_technical_up(1-a),e),_technical_umul(a,rd),rp)
        rm=_technical_uadd(_technical_rho(_technical_uadd(abs(y),affine)),_technical_rho(y))
        alpha_radius=_technical_umul(2.,_technical_umul(_technical_up(alpha_error),_technical_uadd(abs(d),e))) if alpha_error else 0.
        bound[i]=_technical_uadd(affine,rm,alpha_radius)
    return out,bound

def reference_technical_wilder(values, period):
    x=_technical_vector(values)
    if isinstance(period,bool) or not isinstance(period,int) or period<1:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    out=np.full(len(x),np.nan);bound=out.copy()
    if len(x)<period:return out,bound
    out[period-1],bound[period-1]=_technical_seed(x[:period]);k=period-1
    for i in range(period,len(x)):
        m=float(out[i-1]);e=float(bound[i-1]);product=k*m;total=product+float(x[i]);value=total/period;out[i]=value
        if e==0. and m==x[i]:
            from fractions import Fraction
            exact=Fraction.from_float(m)
            if Fraction.from_float(product)==k*exact and Fraction.from_float(total)==period*exact and value==m:
                bound[i]=0.;continue
        carried=_technical_umul(k,e)
        ep=_technical_uadd(_technical_rho(_technical_uadd(abs(product),carried)),_technical_rho(product))
        es=_technical_uadd(_technical_rho(_technical_uadd(abs(total),carried,ep)),_technical_rho(total))
        affine=_technical_udiv(_technical_uadd(carried,ep,es),period)
        ed=_technical_uadd(_technical_rho(_technical_uadd(abs(value),affine)),_technical_rho(value))
        bound[i]=_technical_uadd(affine,ed)
    return out,bound

def _technical_sub(a,b,ea=0.,eb=0.):
    a,b,ea,eb=np.broadcast_arrays(a,b,ea,eb);value=a-b
    valid=np.isfinite(value)&np.isfinite(ea)&np.isfinite(eb);radius=np.full(np.shape(value),np.nan)
    if np.any(valid):
        error=_technical_uadd(ea[valid],eb[valid])
        radius[valid]=_technical_uadd(error,_technical_rho(_technical_uadd(np.abs(value[valid]),error)),_technical_rho(value[valid]))
        exact=valid&(ea==0)&(eb==0)&((a==b)|(b==0));radius[exact]=0.
    return value,radius

def _technical_scale(a,ea,factor):
    a,ea=np.broadcast_arrays(a,ea);value=a*factor;valid=np.isfinite(value)&np.isfinite(ea)
    radius=np.full(np.shape(value),np.nan)
    if np.any(valid):
        error=_technical_umul(ea[valid],abs(factor))
        radius[valid]=_technical_uadd(error,_technical_rho(_technical_uadd(np.abs(value[valid]),error)),_technical_rho(value[valid]))
        if factor in (0.,1.,-1.,2.,-2.):radius[valid]=error
        radius[valid&(a==0)&(ea==0)]=0.
    return value,radius

def _technical_div(a,b,ea=0.,eb=0.):
    a,b,ea,eb=np.broadcast_arrays(a,b,ea,eb);out=np.full(a.shape,np.nan);radius=out.copy()
    finite=np.isfinite(a)&np.isfinite(b)&np.isfinite(ea)&np.isfinite(eb);defined=finite&(b>0)
    if np.any(defined):
        av,bv,ae,be=(v[defined] for v in (a,b,ea,eb))
        lower=np.where(be==0,bv,np.nextafter(bv-be,-np.inf))
        if np.any(lower<=0):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_AMBIGUOUS_DENOMINATOR')
        upper=np.where(be==0,bv,np.nextafter(bv+be,np.inf))
        al=np.where(ae==0,av,np.nextafter(av-ae,-np.inf));ah=np.where(ae==0,av,np.nextafter(av+ae,np.inf))
        candidates=np.stack((al/lower,al/upper,ah/lower,ah/upper))
        lo=np.nextafter(candidates.min(axis=0),-np.inf);hi=np.nextafter(candidates.max(axis=0),np.inf)
        out[defined]=av/bv
        radius[defined]=_technical_up(np.maximum(out[defined]-lo,hi-out[defined]))
        exact=defined&(ea==0)&(eb==0)&((a==0)|(a==b)|(b==1));radius[exact]=0.
    return out,radius

def native_resample_calendar_frame(m5, timeframe):
    durations={'M5':5*MINUTE,'M15':15*MINUTE,'H1':60*MINUTE,'H4':240*MINUTE,'D1':DAY}
    if timeframe not in durations:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    required=('bar_start_ns','open','high','low','close','volume')
    if set(m5)!=set(required):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    t=_projection_integer_clock(m5['bar_start_ns'],'TECHNICAL_NUMERIC_REFERENCE_CLOCK',grid=5*MINUTE)
    if not len(t):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    for name in required[1:]:
        a=np.asarray(m5[name])
        dtype=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if a.dtype!=dtype or a.shape!=t.shape or not np.isfinite(a).all() or np.any(a<=0):
            raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    if (np.any(m5['high']<np.maximum(m5['open'],m5['close'])) or np.any(m5['low']>np.minimum(m5['open'],m5['close']))
            or np.any(m5['high']<m5['low'])):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    duration=durations[timeframe];offset=(22*60*MINUTE)%duration
    labels=((t-offset)//duration)*duration+offset;stamps,first,count=np.unique(labels,return_index=True,return_counts=True)
    keep=(stamps>=t[0])&(stamps+duration<=int(t[-1])+5*MINUTE)
    result={'bar_start_ns':stamps[keep],'open':m5['open'][first][keep],
        'high':np.maximum.reduceat(m5['high'],first)[keep],'low':np.minimum.reduceat(m5['low'],first)[keep],
        'close':m5['close'][first+count-1][keep]}
    quantities=[sum(int(v) for v in m5['volume'][row:row+n]) for row,n in zip(first,count)]
    if any(v>np.iinfo(np.int64).max for v in quantities):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
    result['volume']=np.asarray(quantities,np.int64)[keep]
    return result

def _technical_rolling_sum(values,window):
    """Independent dyadic ring oracle + full-prefix Kahan error-only shadow."""
    from fractions import Fraction
    x=_technical_vector(values)
    if isinstance(window,bool) or not isinstance(window,int) or window<1:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    out=np.full(len(x),np.nan);bound=out.copy();ring=[];exact=Fraction();s=ca=cr=error=0.;same=0;previous=None
    for i,value in enumerate(x):
        v=float(value)
        if i>=window:
            old=ring.pop(0);exact-=old;z=-float(old);y=z-cr;t=s+y;d=t-s;new=d-y
            r1=0. if cr==0. or z==cr else _technical_rho(y)
            r2=0. if t==s or s==0. else _technical_rho(d)
            r3=0. if d==y or y==0. else _technical_rho(new)
            error=_technical_uadd(error,r1,r2,r3);s=t;cr=new
        q=Fraction.from_float(v);ring.append(q);exact+=q;y=v-ca;t=s+y;d=t-s;new=d-y
        r1=0. if ca==0. or v==ca else _technical_rho(y)
        r2=0. if t==s or s==0. else _technical_rho(d)
        r3=0. if d==y or y==0. else _technical_rho(new)
        error=_technical_uadd(error,r1,r2,r3);s=t;ca=new
        same=same+1 if previous==v else 1;previous=v
        if i+1>=window:
            center=float(exact);out[i]=center;ref=float(abs(Fraction.from_float(center)-exact))
            if same>=window:
                # pandas calc_sum bypasses the running state on a uniform
                # window; this output shortcut does not reset its state.
                product=v*window;bound[i]=_technical_uadd(_technical_up(float(abs(Fraction.from_float(product)-exact))),_technical_up(ref))
                if Fraction.from_float(product)==exact and Fraction.from_float(center)==exact:bound[i]=0.
            else:bound[i]=_technical_uadd(error,abs(ca),abs(cr),_technical_up(ref))
    return out,bound

def reference_technical_rolling_vwap(close,volume,window):
    x=_technical_vector(close);v=np.asarray(volume)
    if v.shape!=x.shape or v.ndim!=1 or not np.isfinite(v).all() or np.any(v<=0) or np.any(v!=np.floor(v)):
        raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    v=v.astype(np.float64);product=x*v
    if not np.isfinite(product).all():raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
    numerator,en=_technical_rolling_sum(product,window);denominator,ed=_technical_rolling_sum(v,window)
    # The exact same represented products feed producer and independent
    # window oracle; product RN is shared, never read from the cache.
    return _technical_div(numerator,denominator,en,ed)


def _technical_sqrt_bracket(numerator,denominator):
    """Bracket the real rational sqrt by binary64 numbers using integer squares."""
    import math
    from fractions import Fraction
    if numerator<0 or denominator<=0:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    value=math.sqrt(float(Fraction(numerator,denominator)))
    if not math.isfinite(value):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
    def compare(v):
        p,q=float(v).as_integer_ratio();return p*p*denominator-numerator*q*q
    lower=upper=value
    while compare(lower)>0:lower=math.nextafter(lower,-math.inf)
    while compare(upper)<0:upper=math.nextafter(upper,math.inf)
    return value,lower,upper

def reference_technical_bb20(close):
    """Exact ring moments are the value oracle. The vendor mean/compensation
    shadow is ONLY the full-prefix error ledger (ERROR_SPEC.BB20).

    Z=C+D is exact dyadic: Z_new=Z+s*q, algebraically identical to the
    declared signed-D recurrence, without growing Fraction denominators.
    """
    import math
    from fractions import Fraction
    x=_technical_vector(close)
    if np.any(x<=0):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    nrows=len(x);values={k:np.full(nrows,np.nan) for k in ('mean','variance','std')}
    bounds={k:np.full(nrows,np.nan) for k in values}
    if not nrows:return values,bounds,{'source_rows':0,'full_prefix_updates':0}
    # One extra exponent bit includes represented means/prev_mean near the
    # lowest source value. Every conversion is checked exact below.
    exponent=math.frexp(float(x.min()))[1]-1;k=max(0,53-exponent);D=1<<k;D2=D*D
    def integer(v):
        p,q=float(v).as_integer_ratio()
        if D%q:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_BB_DYADIC_DOMAIN')
        return p*(D//q)
    ring=[];S=Q=Z=0;n=0;mean=ca=cr=R=0.;same=0;previous=None;updates=0;uniform_outputs=0
    mean_sum,mean_error=_technical_rolling_sum(x,20)
    mean_values,mean_bounds=_technical_div(mean_sum,np.full(nrows,20.),mean_error,0.)
    values['mean']=mean_values;bounds['mean']=mean_bounds
    for row,item in enumerate(x):
        v=float(item);vi=integer(v)
        operations=[]
        if len(ring)==20:operations.append((-1,ring.pop(0)))
        operations.append((1,vi))
        for sign,xi in operations:
            actual=float(Fraction(xi,D));comp=ca if sign==1 else cr
            prev=mean-comp;y=actual-comp;t=y-mean;new_comp=(t+mean)-y
            n+=sign
            if n==0:
                # The source's genuinely empty-state branch resets M2/state.
                mean=ca=cr=R=0.;S=Q=Z=0;continue
            new_mean=mean+t/n if sign==1 else mean-t/n
            A=xi-integer(prev);B=xi-integer(new_mean);Z+=sign*A*B
            S+=sign*xi;Q+=sign*xi*xi
            if sign==1:ca=new_comp
            else:cr=new_comp
            mean=new_mean;updates+=1
            # A/B are exact real differences of represented shadow operands.
            # Only their two RN subtractions/product/M2 update enter radius.
            am=0. if A==0 else float(_technical_up(float(Fraction(abs(A),D))))
            bm=0. if B==0 else float(_technical_up(float(Fraction(abs(B),D))))
            if A==0 or B==0:local=0.
            else:
                ra=float(_technical_rho(am));rb=float(_technical_rho(bm))
                local=float(_technical_uadd(_technical_umul(am,rb),_technical_umul(bm,ra),_technical_umul(ra,rb),
                    _technical_rho(_technical_umul(_technical_uadd(am,ra),_technical_uadd(bm,rb)))))
            zm=0. if Z==0 else float(_technical_up(float(Fraction(abs(Z),D2))))
            if local or R:
                total=float(_technical_uadd(R,local));R=float(_technical_uadd(total,_technical_rho(_technical_uadd(zm,total))))
            if not all(math.isfinite(z) for z in (mean,ca,cr,R)):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW')
        ring.append(vi);same=same+1 if previous==v else 1;previous=v
        if n<20:continue
        Cnum=n*Q-S*S
        if Cnum<0:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_BB_EXACT_VARIANCE')
        exact_mean=float(Fraction(S,n*D));old_mean=values['mean'][row]
        values['mean'][row]=exact_mean
        bounds['mean'][row]=_technical_uadd(bounds['mean'][row],_technical_up(abs(old_mean-exact_mean)))
        variance_exact=Fraction(Cnum,n*n*D2);var=float(variance_exact)
        # Cnum/n is unnormalised M2; Cnum/n**2 is population variance.
        # q accumulation Z is M2. D=Z-C has numerator n*Z-Cnum.
        drift=Fraction(abs(n*Z-Cnum),n*n*D2)
        drift_upper=0. if not drift else float(_technical_up(float(drift)))
        ref_error=abs(Fraction.from_float(var)-variance_exact)
        vr=float(_technical_uadd(drift_upper,_technical_udiv(R,n),
            _technical_rho(_technical_udiv(_technical_uadd(zm,R),n)),_technical_up(float(ref_error))))
        values['variance'][row]=var;bounds['variance'][row]=vr
        std,_,_=_technical_sqrt_bracket(Cnum,n*n*D2);values['std'][row]=std
        if same>=n:
            # calc_var uniform shortcut is an OUTPUT only. Z/R are preserved.
            uniform_outputs+=1;values['variance'][row]=0.;values['std'][row]=0.
            bounds['variance'][row]=0.;bounds['std'][row]=0.
            values['mean'][row]=v;bounds['mean'][row]=0.;continue
        lo=max(0.,math.nextafter(var-vr,-math.inf));hi=max(0.,math.nextafter(var+vr,math.inf))
        lop,loq=lo.as_integer_ratio();hip,hiq=hi.as_integer_ratio()
        _,sqrt_lower,_=_technical_sqrt_bracket(lop,loq);_,_,sqrt_upper=_technical_sqrt_bracket(hip,hiq)
        bounds['std'][row]=_technical_up(max(abs(std-sqrt_lower),abs(sqrt_upper-std)))
    return values,bounds,{'source_rows':nrows,'full_prefix_updates':updates,'uniform_shortcuts':uniform_outputs,
        'full_prefix_shadow_not_reset_by_uniform_output':True,'moment_oracle':'exact_dyadic_ring_S_Q','shadow_used_for_values':False}

def reference_technical_block(ohlcv):
    """Twenty-three independent technical formulas on one full observed TF axis."""
    names=('bar_start_ns','open','high','low','close','volume')
    if set(ohlcv)!=set(names):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    time=_projection_integer_clock(ohlcv['bar_start_ns'],'TECHNICAL_NUMERIC_REFERENCE_CLOCK')
    n=len(time)
    for name in names[1:]:
        a=np.asarray(ohlcv[name]);expected=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if a.dtype!=expected or a.shape!=time.shape or not np.isfinite(a).all() or np.any(a<=0):
            raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    high,low,close,open_=(ohlcv[k] for k in ('high','low','close','open'))
    if np.any(high<np.maximum(close,open_)) or np.any(low>np.minimum(close,open_)) or np.any(high<low):
        raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SOURCE')
    out={};error={};tr=high-low
    if n>1:tr[1:]=np.maximum.reduce((tr[1:],np.abs(high[1:]-close[:-1]),np.abs(low[1:]-close[:-1])))
    atr,ea=reference_technical_wilder(tr,14)
    out['atr_bps_14'],error['atr_bps_14']=_technical_scale(*_technical_div(atr,close,ea,0.),1e4)
    def shifted(a,lag):
        v=np.full(n,np.nan)
        if n>lag:v[lag:]=a[:-lag]
        return v
    for lag in (5,20):
        numerator,en=_technical_sub(close,shifted(close,lag))
        out[f'mom_{lag}_atr'],error[f'mom_{lag}_atr']=_technical_div(numerator,atr,en,ea)
    out['close_open_atr'],error['close_open_atr']=_technical_div(*_technical_sub(close,open_)[:1],atr,
        _technical_sub(close,open_)[1],ea)
    changes=np.diff(close);g=np.maximum(changes,0.);l=np.maximum(-changes,0.)
    gain,eg=reference_technical_wilder(g,14);loss,el=reference_technical_wilder(l,14)
    rsi=np.full(n,np.nan);er=rsi.copy()
    for i in range(14,n):
        j=i-1;G,L=float(gain[j]),float(loss[j]);eG,eL=float(eg[j]),float(el[j])
        if L==0. and eL==0.:rsi[i]=50. if G==0. else 100.;er[i]=0.;continue
        if G==0. and eG==0.:rsi[i]=0.;er[i]=0.;continue
        strength,es=_technical_div(G,L,eG,eL);den=1.+strength;ed=_technical_uadd(es,_technical_rho(_technical_uadd(abs(den),es)),_technical_rho(den))
        fraction,ef=_technical_div(100.,den,0.,ed);rsi[i]=100.-fraction;er[i]=_technical_uadd(ef,_technical_rho(_technical_uadd(abs(rsi[i]),ef)),_technical_rho(rsi[i]))
    out['rsi14_centered'],error['rsi14_centered']=_technical_div(*_technical_sub(rsi,50.,er,0.)[:1],50.,
        _technical_sub(rsi,50.,er,0.)[1],0.)
    emas={span:reference_technical_ema(close,span) for span in (20,50,100,200)}
    for span,(ema,ee) in emas.items():
        gap,ep=_technical_sub(close,ema,0.,ee)
        out[f'ema{span}_dist_atr'],error[f'ema{span}_dist_atr']=_technical_div(gap,atr,ep,ea)
        lag=5 if span in (20,50) else 20
        slope,es=_technical_sub(ema,shifted(ema,lag),ee,shifted(ee,lag))
        out[f'ema{span}_slope_atr'],error[f'ema{span}_slope_atr']=_technical_div(slope,atr,es,ea)
    stack=np.full(n,np.nan);age=stack.copy()
    for i in range(199,n):
        state=0;comparisons=[]
        for a,b in ((20,50),(50,100),(100,200)):
            difference=emas[a][0][i]-emas[b][0][i];uncertainty=0. if emas[a][1][i]==emas[b][1][i]==0. else _technical_uadd(emas[a][1][i],emas[b][1][i])
            if abs(difference)<=uncertainty and not (difference==0. and uncertainty==0.):
                raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_AMBIGUOUS_STACK')
            comparisons.append(np.sign(difference))
        if all(x>0 for x in comparisons):state=1
        elif all(x<0 for x in comparisons):state=-1
        stack[i]=state;age[i]=0. if i==199 or stack[i]!=stack[i-1] else age[i-1]+1
    out['ema_stack_aligned_v2']=stack;error['ema_stack_aligned_v2']=np.where(np.isfinite(stack),0.,np.nan)
    out['trend_state_age_bars']=age;error['trend_state_age_bars']=np.where(np.isfinite(age),0.,np.nan)
    for window,name in ((5,'vwap_rolling5_dist_atr'),(20,'vwap20_dist_atr'),(96,'vwap96_dist_atr')):
        vwap,ev=reference_technical_rolling_vwap(close,ohlcv['volume'],window)
        gap,ep=_technical_sub(close,vwap,0.,ev);out[name],error[name]=_technical_div(gap,atr,ep,ea)
        if window==5:
            difference,ed=_technical_sub(vwap,shifted(vwap,5),ev,shifted(ev,5))
            out['vwap_rolling5_slope_atr'],error['vwap_rolling5_slope_atr']=_technical_div(difference,atr,ed,ea)
    bb,eb,bb_report=reference_technical_bb20(close)
    two_std,et=_technical_scale(bb['std'],eb['std'],2.)
    upper,eu=_technical_sub(bb['mean'],-two_std,eb['mean'],et)
    lower,elo=_technical_sub(bb['mean'],two_std,eb['mean'],et)
    width,ew=_technical_sub(upper,lower,eu,elo)
    # Identical uniform bands are an exact zero-width observation, not epsilon.
    uniform=np.isfinite(bb['std'])&(bb['std']==0)&(eb['std']==0)&(eb['mean']==0)
    width[uniform]=0.;ew[uniform]=0.
    numerator,en=_technical_sub(close,lower,0.,elo)
    out['bb_position'],error['bb_position']=_technical_div(numerator,width,en,ew)
    out['bb_width_atr'],error['bb_width_atr']=_technical_div(width,atr,ew,ea)
    # ADX uses rows1..14 (unlike ATR's TR0 seed), independent Wilder sums.
    plus=np.zeros(n);minus=np.zeros(n)
    if n>1:
        up=high[1:]-high[:-1];down=low[:-1]-low[1:]
        plus[1:]=np.where((up>down)&(up>0),up,0.);minus[1:]=np.where((down>up)&(down>0),down,0.)
    smooth=[]
    for source in (tr,plus,minus):
        values=np.full(n,np.nan);ebound=values.copy()
        if n>14:
            values[14],ebound[14]=_technical_seed(source[1:15],divide=False)
            for i in range(15,n):
                old=float(values[i-1]);e=float(ebound[i-1]);q=old/14.;difference=old-q;value=difference+source[i];values[i]=value
                if old==0. and source[i]==0. and e==0.:ebound[i]=0.;continue
                carried=_technical_umul(_technical_up(13./14.),e)
                rq=_technical_uadd(_technical_rho(_technical_uadd(abs(q),_technical_udiv(e,14.))),_technical_rho(q))
                rs=_technical_uadd(_technical_rho(_technical_uadd(abs(difference),carried,rq)),_technical_rho(difference))
                ra=_technical_uadd(_technical_rho(_technical_uadd(abs(value),carried,rq,rs)),_technical_rho(value))
                ebound[i]=_technical_uadd(carried,rq,rs,ra)
        smooth.append((values,ebound))
    p,ep=_technical_scale(*_technical_div(smooth[1][0],smooth[0][0],smooth[1][1],smooth[0][1]),100.)
    m,em=_technical_scale(*_technical_div(smooth[2][0],smooth[0][0],smooth[2][1],smooth[0][1]),100.)
    difference,ed=_technical_sub(p,m,ep,em);total,etotal=_technical_sub(p,-m,ep,em)
    signed,esigned=_technical_div(difference,total,ed,etotal)
    out['di_spread_signed']=signed;error['di_spread_signed']=esigned
    dx,edx=_technical_scale(*_technical_div(np.abs(difference),total,ed,etotal),100.)
    adx=np.full(n,np.nan);eax=adx.copy()
    if n>27 and np.isfinite(dx[14:28]).all():
        end=28
        while end<n and np.isfinite(dx[end]):end+=1
        adx_part,part_error=reference_technical_wilder(dx[14:end],14)
        adx[14:end]=adx_part
        # Account for uncertainty in every upstream DX, through seed/recurrence.
        carry=float(np.max(edx[14:28]));eax[27]=_technical_uadd(part_error[13],carry)
        for i in range(28,end):carry=_technical_udiv(_technical_uadd(_technical_umul(13.,carry),edx[i]),14.);eax[i]=_technical_uadd(part_error[i-14],carry)
    out['adx14']=adx;error['adx14']=eax
    out={name:out[name] for name in TECHNICAL_NUMERIC_FIELDS};error={name:error[name] for name in TECHNICAL_NUMERIC_FIELDS}
    first={name:int(np.flatnonzero(np.isfinite(value))[0]) if np.isfinite(value).any() else None for name,value in out.items()}
    return out,error,{'source_rows':n,'first_valid_rows':first,'full_history':True,'gap_or_session_reset':False,'BB20':bb_report}


def compare_technical_reference(reference,actual,bounds,*,discrete_fields=('ema_stack_aligned_v2','trend_state_age_bars')):
    if set(reference)!=set(actual) or set(reference)!=set(bounds):raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    report={}
    for name,value in reference.items():
        q=np.asarray(value);a=np.asarray(actual[name]);b=np.asarray(bounds[name])
        if (q.dtype!=np.dtype('float64') or a.dtype!=np.dtype('float32') or b.dtype!=np.dtype('float64')
                or q.ndim!=1 or q.shape!=a.shape or q.shape!=b.shape or np.isinf(q).any() or np.isinf(a).any()
                or np.isinf(b).any() or np.any(b[np.isfinite(b)]<0)):
            raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA:'+name)
        finite=np.isfinite(q)
        if not np.array_equal(np.isnan(q),np.isnan(a)) or not np.array_equal(np.isnan(q),np.isnan(b)):
            raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_NAN_MASK:'+name)
        if name in discrete_fields:
            if not np.array_equal(q[finite],a[finite].astype(np.float64)):
                raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_DISCRETE_MISMATCH:'+name)
            total=np.zeros(int(finite.sum()))
        else:
            center=q[finite];radius=b[finite]
            lo=np.where(radius==0.,center,np.nextafter(center-radius,-np.inf))
            hi=np.where(radius==0.,center,np.nextafter(center+radius,np.inf))
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                lo32=lo.astype(np.float32);hi32=hi.astype(np.float32)
            if not np.isfinite(lo32).all() or not np.isfinite(hi32).all():
                raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_OVERFLOW:'+name)
            if np.any(a[finite]<lo32) or np.any(a[finite]>hi32):
                raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_VALUE_MISMATCH:'+name)
            total=np.maximum(np.abs(center-lo32.astype(np.float64)),np.abs(hi32.astype(np.float64)-center))
        delta=np.abs(a[finite].astype(np.float64)-q[finite])
        report[name]={'finite_published_values_compared':int(finite.sum()),'nan_masked_values_compared':int((~finite).sum()),
            'maximum_absolute_error':float(delta.max()) if len(delta) else 0.,
            'maximum_forward_roundoff_bound':float(total.max()) if len(total) else 0.,
            'maximum_float64_radius':float(b[finite].max()) if finite.any() else 0.,
            'numeric_reference_qualified':bool(finite.any()),'nonconstant_reference':bool(finite.any() and np.ptp(q[finite])>0),
            'minimum_reference':float(q[finite].min()) if finite.any() else None,'maximum_reference':float(q[finite].max()) if finite.any() else None,
            'discrete_exact':name in discrete_fields}
    return report

def _technical_native_footer(path,seal):
    file=pq.ParquetFile(path);schema=file.schema_arrow
    expected={'time':pa.timestamp('ns',tz='UTC'),'open':pa.float64(),'high':pa.float64(),'low':pa.float64(),
        'close':pa.float64(),'volume':pa.int64()}
    if any(name not in schema.names or schema.field(name).type!=dtype for name,dtype in expected.items()):
        raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
    for i in range(file.metadata.num_row_groups):
        row=file.metadata.row_group(i);column=[row.column(j) for j in range(row.num_columns) if row.column(j).path_in_schema=='time']
        if len(column)!=1:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
        stat=column[0].statistics
        if stat is None or not stat.has_min_max or stat.null_count!=0:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_SCHEMA')
        if pd.Timestamp(stat.max).value>=seal:raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_TEST_ROW_GROUP')
    return file.metadata.num_rows

def _technical_npy_header(path,shape,dtype):
    with Path(path).open('rb') as handle:
        version=np.lib.format.read_magic(handle);actual,fortran,observed=np.lib.format._read_array_header(handle,version)
        offset=handle.tell()
    if actual!=shape or fortran or observed!=np.dtype(dtype) or Path(path).stat().st_size!=offset+np.prod(shape)*np.dtype(dtype).itemsize:
        raise RuntimeError('TECHNICAL_NUMERIC_REFERENCE_NPY_HEADER')


TREND_EVENT_NUMERIC_PLAN_ID='NATIVE_M5_TREND_EVENT_NUMERIC_REFERENCE_20261021_128'
TREND_EVENT_NUMERIC_POLICY_KEY='trend_event_numeric_reference_128'
TREND_EVENT_NUMERIC_FIELDS=('ema20_50_bull_state','ema20_50_cross_up','ema20_50_cross_down','ema20_50_state_age_bars',
    'ema50_200_bull_state','ema50_200_cross_up','ema50_200_cross_down','ema50_200_state_age_bars',
    'price_x_ema20_cross_up','price_x_ema20_cross_down','price_x_ema50_cross_up','price_x_ema50_cross_down',
    'price_x_ema200_cross_up','price_x_ema200_cross_down','price_vs_ema20_state_age_bars',
    'price_vs_ema50_state_age_bars','price_vs_ema200_state_age_bars')


def reference_trend_events_from_spread(spread,radius):
    """Certify three-way signs before independent closed-row events and raw ages."""
    x=np.asarray(spread);error=np.asarray(radius)
    if (x.dtype!=np.dtype('float64') or error.dtype!=np.dtype('float64') or x.ndim!=1
            or error.shape!=x.shape or len(x)>=2**24 or np.isinf(x).any() or np.isinf(error).any()
            or not np.array_equal(np.isnan(x),np.isnan(error)) or np.any(error[np.isfinite(error)]<0)):
        raise RuntimeError('TREND_EVENT_NUMERIC_SPREAD_SCHEMA')
    valid=np.isfinite(x);rows=np.flatnonzero(valid)
    state=np.full(len(x),np.nan);up=state.copy();down=state.copy();age=state.copy()
    if not len(rows):return state,up,down,age
    first=int(rows[0])
    if not valid[first:].all():raise RuntimeError('TREND_EVENT_NUMERIC_NAN_SUFFIX')
    center=x[first:];e=error[first:]
    # Frozen ERROR_SPEC128 requires three-way certification even if a
    # particular Boolean conjunction alone could admit a weaker proof.
    with np.errstate(over='ignore',invalid='ignore'):
        lower=np.where(e==0.,center,np.nextafter(center-e,-np.inf))
        upper=np.where(e==0.,center,np.nextafter(center+e,np.inf))
    certified=(lower>0.)|(upper<0.)|((center==0.)&(e==0.))
    certified &= np.isfinite(lower)&np.isfinite(upper)
    if not certified.all():
        local=int(np.flatnonzero(~certified)[0]);row=first+local
        raise RuntimeError('TREND_EVENT_BRANCH_AMBIGUOUS:row='+str(row)+':center='+repr(float(center[local]))
            +':radius='+repr(float(e[local]))+':interval='+repr((float(lower[local]),float(upper[local]))))
    positive=center>0.;state[first:]=positive.astype(np.float64)
    if len(center)>1:
        up[first+1:]=((center[1:]>0.)&(center[:-1]<=0.)).astype(np.float64)
        down[first+1:]=((center[1:]<0.)&(center[:-1]>=0.)).astype(np.float64)
    # Independent observed-row run starts; no calendar/session reset.
    positions=np.arange(len(center),dtype=np.int64);changed=np.ones(len(center),dtype=bool)
    changed[1:]=positive[1:]!=positive[:-1]
    starts=np.maximum.accumulate(np.where(changed,positions,0))
    age[first:]=(positions-starts).astype(np.float64)
    return state,up,down,age


def reference_trend_event_block(ohlcv):
    """Seventeen independent discrete EMA events on one full historical TF axis."""
    names=('bar_start_ns','open','high','low','close','volume')
    if set(ohlcv)!=set(names):raise RuntimeError('TREND_EVENT_NUMERIC_SOURCE_SCHEMA')
    clock=_projection_integer_clock(ohlcv['bar_start_ns'],'TREND_EVENT_NUMERIC_SOURCE_CLOCK')
    if len(clock)>=2**24:raise RuntimeError('TREND_EVENT_NUMERIC_AGE_STORAGE')
    for name in names[1:]:
        a=np.asarray(ohlcv[name]);dtype=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if a.dtype!=dtype or a.shape!=clock.shape or not np.isfinite(a).all() or np.any(a<=0):
            raise RuntimeError('TREND_EVENT_NUMERIC_SOURCE_SCHEMA')
    high,low,close,open_=(ohlcv[key] for key in ('high','low','close','open'))
    if np.any(high<np.maximum(close,open_)) or np.any(low>np.minimum(close,open_)) or np.any(high<low):
        raise RuntimeError('TREND_EVENT_NUMERIC_SOURCE_SCHEMA')
    emas={span:reference_technical_ema(close,span) for span in (20,50,200)}
    out={}
    def events(a,b,ea,eb,label):
        spread,radius=_technical_sub(a,b,ea,eb)
        try:return reference_trend_events_from_spread(spread,radius)
        except RuntimeError as exc:raise RuntimeError(str(exc)+':field='+label) from exc
    for fast,slow in ((20,50),(50,200)):
        state,up,down,age=events(emas[fast][0],emas[slow][0],emas[fast][1],emas[slow][1],f'ema{fast}_{slow}')
        base=f'ema{fast}_{slow}_'
        out[base+'bull_state']=state;out[base+'cross_up']=up;out[base+'cross_down']=down;out[base+'state_age_bars']=age
    for span in (20,50,200):
        _,up,down,age=events(close,emas[span][0],0.,emas[span][1],f'price_x_ema{span}')
        out[f'price_x_ema{span}_cross_up']=up;out[f'price_x_ema{span}_cross_down']=down
        out[f'price_vs_ema{span}_state_age_bars']=age
    out={name:out[name] for name in TREND_EVENT_NUMERIC_FIELDS}
    bounds={name:np.where(np.isfinite(value),0.,np.nan) for name,value in out.items()}
    first={name:int(np.flatnonzero(np.isfinite(value))[0]) if np.isfinite(value).any() else None for name,value in out.items()}
    return out,bounds,{'source_rows':len(clock),'first_valid_rows':first,'EMA_spans':[20,50,200],
        'full_history':True,'gap_or_session_reset':False,'three_way_sign_certification':True,
        'ambiguous_branches':0,'BB_or_ADX_or_other_technical_formulas_recomputed':False}


NATIVE_SMC_NUMERIC_PLAN_ID='NATIVE_M5_NATIVE_SMC_NUMERIC_REFERENCE_20261021_129'
NATIVE_SMC_NUMERIC_POLICY_KEY='native_smc_numeric_reference_129'
NATIVE_SMC_NUMERIC_FIELDS=('mtf_smc_structure_bias', 'mtf_smc_bos_up', 'mtf_smc_bos_down', 'mtf_smc_choch_up', 'mtf_smc_choch_down', 'mtf_smc_sweep_up_state', 'mtf_smc_sweep_down_state', 'mtf_smc_sweep_up_depth_atr', 'mtf_smc_sweep_down_depth_atr', 'mtf_smc_pivot_envelope_position', 'mtf_smc_range_width_atr', 'mtf_smc_bos_displacement_atr', 'mtf_smc_sweep_up_event', 'mtf_smc_sweep_down_event', 'mtf_smc_sweep_event_age_bars', 'mtf_smc_swing_state', 'mtf_smc_sweep_last_event_side', 'mtf_geometry_support_dist_atr', 'mtf_geometry_resistance_dist_atr', 'mtf_geometry_support_age_bars', 'mtf_geometry_resistance_age_bars', 'mtf_geometry_support_rail_slope_atr_per_bar', 'mtf_geometry_resistance_rail_slope_atr_per_bar', 'mtf_geometry_range_mid_dist_atr')
NATIVE_SMC_DISCRETE_FIELDS=('mtf_smc_structure_bias', 'mtf_smc_bos_up', 'mtf_smc_bos_down', 'mtf_smc_choch_up', 'mtf_smc_choch_down', 'mtf_smc_sweep_up_state', 'mtf_smc_sweep_down_state', 'mtf_smc_sweep_up_event', 'mtf_smc_sweep_down_event', 'mtf_smc_sweep_event_age_bars', 'mtf_smc_swing_state', 'mtf_smc_sweep_last_event_side', 'mtf_geometry_support_age_bars', 'mtf_geometry_resistance_age_bars')
NATIVE_SMC_PIVOT_TOLERANCE=1e-12  # smc_v1._detect_swing_pivots source502/504


def reference_native_smc_replay(high,low,close,*,swing_lookback=3):
    """Independent chronological native-price replay, before output warmup."""
    from fractions import Fraction
    h,l,c=(np.asarray(value) for value in (high,low,close));n=len(h)
    if (isinstance(swing_lookback,bool) or not isinstance(swing_lookback,int) or swing_lookback<1
            or n>=2**24 or any(a.dtype!=np.dtype('float64') or a.ndim!=1 or a.shape!=h.shape
                or not np.isfinite(a).all() or np.any(a<=0.) for a in (h,l,c))
            or np.any(h<l) or np.any(h<c) or np.any(l>c)):
        raise RuntimeError('NATIVE_SMC_NUMERIC_REPLAY_SCHEMA')
    k=swing_lookback;high_pivots=np.zeros(n,bool);low_pivots=np.zeros(n,bool)
    if n>=2*k+1:
        tolerance=Fraction.from_float(NATIVE_SMC_PIVOT_TOLERANCE)
        for values,is_high,pivots in ((h,True,high_pivots),(l,False,low_pivots)):
            windows=np.lib.stride_tricks.sliding_window_view(values,2*k+1)
            extremes=windows.max(axis=1) if is_high else windows.min(axis=1)
            unique,inverse=np.unique(extremes,return_inverse=True)
            # Exact dyadic arithmetic followed by one RN64 conversion reproduces
            # the source's represented max-minus/min-plus tolerance threshold.
            thresholds=np.array([float(Fraction.from_float(float(v))+(-tolerance if is_high else tolerance))
                for v in unique],dtype=np.float64)[inverse]
            pivots[k:n-k]=(values[k:n-k]>=thresholds if is_high else values[k:n-k]<=thresholds)
    indices=('last_high_idx','prev_high_idx','last_low_idx','prev_low_idx')
    prices=('last_high_price','prev_high_price','last_low_price','prev_low_price')
    roots=('structure_bias','structure_available','swing_state','bos_up','bos_down','choch_up','choch_down',
        'sweep_up','sweep_down','sweep_up_event','sweep_down_event','sweep_event_age_bars','sweep_last_event_side')
    out={name:np.full(n,-1,np.int64) for name in indices}
    out.update({name:np.full(n,np.nan) for name in prices+roots})
    last={'high':None,'low':None};previous={'high':None,'low':None}
    consumed_bos={'high':None,'low':None};consumed_sweep={'high':None,'low':None}
    regime=0;last_sweep=None;last_side=np.nan
    for row in range(n):
        candidate=row-k
        if candidate>=k:
            for side,values,pivots in (('high',h,high_pivots),('low',l,low_pivots)):
                if pivots[candidate]:previous[side],last[side]=last[side],(candidate,float(values[candidate]))
        for side in ('high','low'):
            for prefix,entry in (('last',last[side]),('prev',previous[side])):
                if entry is not None:
                    out[prefix+'_'+side+'_idx'][row]=entry[0];out[prefix+'_'+side+'_price'][row]=entry[1]
        signs=[]
        for side in ('high','low'):
            a,b=last[side],previous[side]
            signs.append(0 if a is None or b is None else int(a[1]>b[1])-int(a[1]<b[1]))
        hs,ls=signs;bias=(hs+ls)/2.;state={(1,1):0,(1,-1):1,(-1,1):2,(-1,-1):3}.get((hs,ls),4)
        levels=[entry[1] for entry in (*last.values(),*previous.values()) if entry is not None]
        structure_available=len(levels)==4 and max(levels)>min(levels)
        out['structure_bias'][row]=bias;out['swing_state'][row]=state;out['structure_available'][row]=float(structure_available)
        bos={}
        for side in ('high','low'):
            entry=last[side];identity=None if entry is None else entry[0]
            fired=entry is not None and (c[row]>entry[1] if side=='high' else c[row]<entry[1]) and consumed_bos[side]!=identity
            bos[side]=bool(fired)
            if fired:consumed_bos[side]=identity
            out['bos_'+('up' if side=='high' else 'down')][row]=float(fired)
        choch_up=regime<0 and bos['high'];choch_down=regime>0 and bos['low']
        if choch_up:regime=1
        elif choch_down:regime=-1
        elif regime==0 and structure_available and bias!=0.:regime=1 if bias>0 else -1
        out['choch_up'][row]=float(choch_up);out['choch_down'][row]=float(choch_down)
        events={}
        for side,wick in (('high',h[row]),('low',l[row])):
            entry=last[side];identity=None if entry is None else entry[0]
            swept=entry is not None and consumed_bos[side]!=identity and (
                wick>entry[1] and c[row]<=entry[1] if side=='high' else wick<entry[1] and c[row]>=entry[1])
            event=swept and consumed_sweep[side]!=identity
            if event:consumed_sweep[side]=identity
            suffix='up' if side=='high' else 'down';out['sweep_'+suffix][row]=float(swept)
            out['sweep_'+suffix+'_event'][row]=float(event);events[side]=bool(event)
        if any(events.values()):
            last_sweep=row;last_side=0. if all(events.values()) else 1. if events['high'] else -1.
        out['sweep_event_age_bars'][row]=np.nan if last_sweep is None else row-last_sweep
        out['sweep_last_event_side'][row]=last_side
    return out


def reference_native_smc_block(ohlcv):
    """Independent native spot SMC17 plus its seven same-owner geometry roots."""
    names=('bar_start_ns','open','high','low','close','volume')
    if set(ohlcv)!=set(names):raise RuntimeError('NATIVE_SMC_NUMERIC_SOURCE_SCHEMA')
    clock=_projection_integer_clock(ohlcv['bar_start_ns'],'NATIVE_SMC_NUMERIC_SOURCE_CLOCK');n=len(clock)
    for name in names[1:]:
        a=np.asarray(ohlcv[name]);dtype=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if a.dtype!=dtype or a.shape!=clock.shape or not np.isfinite(a).all() or np.any(a<=0):
            raise RuntimeError('NATIVE_SMC_NUMERIC_SOURCE_SCHEMA')
    high,low,close,open_=(ohlcv[key] for key in ('high','low','close','open'))
    if np.any(high<np.maximum(close,open_)) or np.any(low>np.minimum(close,open_)) or np.any(high<low):
        raise RuntimeError('NATIVE_SMC_NUMERIC_SOURCE_SCHEMA')
    tr=high-low
    if n>1:tr[1:]=np.maximum.reduce((tr[1:],np.abs(high[1:]-close[:-1]),np.abs(low[1:]-close[:-1])))
    atr,ea=reference_technical_wilder(tr,14)
    if np.any(atr[np.isfinite(atr)]<=0.):raise RuntimeError('NATIVE_SMC_NUMERIC_NONPOSITIVE_ATR')
    replay=reference_native_smc_replay(high,low,close)
    lh,ph,ll,pl=(replay[name] for name in ('last_high_price','prev_high_price','last_low_price','prev_low_price'))
    hi,phi,li,pli=(replay[name] for name in ('last_high_idx','prev_high_idx','last_low_idx','prev_low_idx'))
    available=(hi>=0)&(phi>=0)&(li>=0)&(pli>=0)&np.isfinite(atr)
    top=np.maximum.reduce((lh,ph,ll,pl));bottom=np.minimum.reduce((lh,ph,ll,pl))
    width,ew=_technical_sub(top,bottom);out={};radii={}
    for suffix in ('structure_bias','bos_up','bos_down','choch_up','choch_down','sweep_up_event',
            'sweep_down_event','sweep_event_age_bars','swing_state','sweep_last_event_side'):
        out['mtf_smc_'+suffix]=replay[suffix]
    for side in ('up','down'):out['mtf_smc_sweep_'+side+'_state']=replay['sweep_'+side]
    def ratio(name,a,b,ea_=0.,eb_=0.):out[name],radii[name]=_technical_div(a,b,ea_,eb_)
    for side,first,second in (('up',high,lh),('down',ll,low)):
        diff,ed=_technical_sub(first,second);ratio('mtf_smc_sweep_'+side+'_depth_atr',diff,atr,ed,ea)
        name='mtf_smc_sweep_'+side+'_depth_atr';inactive=replay['sweep_'+side]==0.
        out[name][inactive]=0.;radii[name][inactive]=0.
    diff,ed=_technical_sub(close,bottom)
    ratio('mtf_smc_pivot_envelope_position',diff,np.where(available,width,np.nan),ed,ew)
    collapsed=available&(width==0.)
    out['mtf_smc_pivot_envelope_position'][collapsed]=0.;radii['mtf_smc_pivot_envelope_position'][collapsed]=0.
    ratio('mtf_smc_range_width_atr',width,atr,ew,ea)
    selected_high=(replay['bos_up']>0.)&((replay['bos_down']<=0.)|(hi>=li))
    selected_low=(replay['bos_down']>0.)&~selected_high
    displacement=np.zeros(n);edisplacement=np.zeros(n)
    for selected,level in ((selected_high,lh),(selected_low,ll)):
        diff,ed=_technical_sub(close,level);value,error=_technical_div(diff,atr,ed,ea)
        displacement[selected]=value[selected];edisplacement[selected]=error[selected]
    out['mtf_smc_bos_displacement_atr']=displacement;radii['mtf_smc_bos_displacement_atr']=edisplacement
    for side,a,b,index,previous_index in (('support',ll,pl,li,pli),('resistance',lh,ph,hi,phi)):
        diff,ed=_technical_sub(close,a) if side=='support' else _technical_sub(a,close)
        ratio('mtf_geometry_'+side+'_dist_atr',diff,atr,ed,ea)
        out['mtf_geometry_'+side+'_age_bars']=(np.arange(n,dtype=np.int64)-index).astype(np.float64)
        span=(index-previous_index).astype(np.float64);valid=available&(span>0.)
        denominator=np.full(n,np.nan);edenominator=denominator.copy()
        denominator[valid]=span[valid]*atr[valid]
        # ERROR_SPEC129: every positive uncertainty node rounds outward.
        propagated=_technical_umul(span[valid],ea[valid])
        edenominator[valid]=_technical_uadd(propagated,
            _technical_rho(_technical_uadd(np.abs(denominator[valid]),propagated)),_technical_rho(denominator[valid]))
        identity=valid&(span==1.);edenominator[identity]=ea[identity]
        diff,ed=_technical_sub(a,b)
        ratio('mtf_geometry_'+side+'_rail_slope_atr_per_bar',diff,denominator,ed,edenominator)
    total,et=_technical_sub(top,-bottom);mid,em=_technical_scale(total,et,.5)
    diff,ed=_technical_sub(close,mid,0.,em);ratio('mtf_geometry_range_mid_dist_atr',diff,atr,ed,ea)
    for name in NATIVE_SMC_NUMERIC_FIELDS:
        value=np.asarray(out[name],dtype=np.float64)
        bound=np.where(np.isfinite(value),0.,np.nan) if name in NATIVE_SMC_DISCRETE_FIELDS else radii[name]
        out[name]=np.where(available,value,np.nan);radii[name]=np.where(available,bound,np.nan)
    out={name:out[name] for name in NATIVE_SMC_NUMERIC_FIELDS};radii={name:radii[name] for name in NATIVE_SMC_NUMERIC_FIELDS}
    first={name:int(np.flatnonzero(np.isfinite(value))[0]) if np.isfinite(value).any() else None for name,value in out.items()}
    return out,radii,{'source_rows':n,'first_valid_rows':first,
        'common_first_valid_row':int(np.flatnonzero(available)[0]) if available.any() else None,
        'full_history':True,'gap_or_session_reset':False,'swing_lookback':3,'pivot_comparison_tolerance':NATIVE_SMC_PIVOT_TOLERANCE,
        'pivot_thresholds_use_exact_dyadic_then_RN64':True,'replay_before_output_warmup':True,
        'positive_witnesses':{name:int(np.sum(value>0.)) for name,value in out.items()},
        'BB_or_ADX_or_other_technical_formulas_recomputed':False}


NATIVE_CORE_SWING_FIELDS=('swing_dist_last_swing_high_atr', 'swing_dist_last_swing_low_atr', 'swing_bars_since_swing_high', 'swing_bars_since_swing_low', 'swing_retracement_from_last_impulse', 'swing_impulse_present', 'swing_high_break_event', 'swing_low_break_event', 'swing_high_break_displacement_atr', 'swing_low_break_displacement_atr', 'bars_since_swing_high_break', 'bars_since_swing_low_break', 'swing_high_sequence_delta_atr', 'swing_low_sequence_delta_atr', 'consecutive_higher_lows_count', 'consecutive_lower_highs_count', 'consecutive_higher_highs_count', 'consecutive_lower_lows_count', 'swing_high_level_intact', 'swing_low_level_intact')
NATIVE_CORE_SWING_DISCRETE_FIELDS=('swing_bars_since_swing_high', 'swing_bars_since_swing_low', 'swing_impulse_present', 'swing_high_break_event', 'swing_low_break_event', 'bars_since_swing_high_break', 'bars_since_swing_low_break', 'consecutive_higher_lows_count', 'consecutive_lower_highs_count', 'consecutive_higher_highs_count', 'consecutive_lower_lows_count', 'swing_high_level_intact', 'swing_low_level_intact')


def reference_native_core_swing(frame,*,atr,atr_radius):
    """Strict lookback-two structure, independent of tolerant SMC pivots."""
    required=('bar_start_ns','high','low','close')
    if any(name not in frame for name in required):raise RuntimeError('NATIVE_CORE_SWING_SOURCE_SCHEMA')
    clock=_projection_integer_clock(frame['bar_start_ns'],'NATIVE_CORE_SWING_CLOCK');n=len(clock)
    h,l,c=(np.asarray(frame[name]) for name in ('high','low','close'))
    if (n>=2**24 or any(a.dtype!=np.dtype('float64') or a.shape!=clock.shape or not np.isfinite(a).all()
            or np.any(a<=0.) for a in (h,l,c)) or np.any(h<l) or np.any(h<c) or np.any(l>c)):
        raise RuntimeError('NATIVE_CORE_SWING_SOURCE_SCHEMA')
    a=np.asarray(atr);ea=np.asarray(atr_radius)
    if (a.dtype!=np.dtype('float64') or ea.dtype!=np.dtype('float64') or a.shape!=clock.shape or ea.shape!=clock.shape
            or np.isinf(a).any() or np.isinf(ea).any() or not np.array_equal(np.isnan(a),np.isnan(ea))
            or np.any(a[np.isfinite(a)]<0.) or np.any(ea[np.isfinite(ea)]<0.)):
        raise RuntimeError('NATIVE_CORE_SWING_ATR_SCHEMA')
    finite=np.isfinite(a);observed=np.flatnonzero(finite)
    if len(observed) and not finite[int(observed[0]):].all():raise RuntimeError('NATIVE_CORE_SWING_ATR_SUFFIX')
    positive=finite&(a>0.)
    lower=np.where(ea==0.,a,np.nextafter(a-ea,-np.inf))
    if np.any(positive&(lower<=0.)) or np.any(finite&(a==0.)&(ea>0.)):
        raise RuntimeError('NATIVE_CORE_SWING_ATR_DOMAIN_AMBIGUOUS')
    denominator=np.where(positive,a,np.nan);edenominator=np.where(positive,ea,np.nan)
    high_pivot=np.zeros(n,bool);low_pivot=np.zeros(n,bool)
    if n>=5:
        for values,pivots,is_high in ((h,high_pivot,True),(l,low_pivot,False)):
            windows=np.lib.stride_tricks.sliding_window_view(values,5)
            neighbours=np.column_stack((windows[:,:2],windows[:,3:]))
            pivots[2:n-2]=(values[2:n-2]>neighbours.max(axis=1) if is_high else values[2:n-2]<neighbours.min(axis=1))
    latest={side:np.full(n,np.nan) for side in ('high','low')}
    previous={side:np.full(n,np.nan) for side in ('high','low')}
    indices={side:np.full(n,-1,np.int64) for side in ('high','low')}
    out={name:np.zeros(n) for name in NATIVE_CORE_SWING_FIELDS}
    for side in ('high','low'):
        out['swing_'+side+'_level_intact'].fill(np.nan)
        out['bars_since_swing_'+side+'_break'].fill(np.nan)
    last={'high':None,'low':None};prev={'high':None,'low':None};armed={'high':False,'low':False}
    last_break={'high':None,'low':None};runs={'higher_highs':0,'lower_highs':0,'higher_lows':0,'lower_lows':0}
    for row in range(n):
        candidate=row-2
        for side,values,pivots in (('high',h,high_pivot),('low',l,low_pivot)):
            if candidate>=2 and pivots[candidate]:
                price=float(values[candidate]);old=last[side]
                if old is not None:
                    greater='higher_highs' if side=='high' else 'higher_lows';less='lower_highs' if side=='high' else 'lower_lows'
                    runs[greater]=runs[greater]+1 if price>old[1] else 0
                    runs[less]=runs[less]+1 if price<old[1] else 0
                prev[side]=old;last[side]=(candidate,price);armed[side]=True
            current=last[side]
            if current is not None:
                indices[side][row]=current[0];latest[side][row]=current[1]
                if prev[side] is not None:previous[side][row]=prev[side][1]
            fired=armed[side] and (c[row]>current[1] if side=='high' else c[row]<current[1])
            if fired:armed[side]=False;last_break[side]=row
            out['swing_'+side+'_break_event'][row]=float(fired)
            if current is not None:out['swing_'+side+'_level_intact'][row]=float(armed[side])
            if last_break[side] is not None:out['bars_since_swing_'+side+'_break'][row]=row-last_break[side]
        for key,value in runs.items():out['consecutive_'+key+'_count'][row]=value
    radii={}
    for side in ('high','low'):
        name='swing_dist_last_swing_'+side+'_atr';difference,ed=_technical_sub(c,latest[side])
        out[name],radii[name]=_technical_div(difference,denominator,ed,edenominator)
        out['swing_bars_since_swing_'+side]=np.where(indices[side]>=0,np.arange(n)-indices[side],np.nan)
        name='swing_'+side+'_break_displacement_atr';fired=out['swing_'+side+'_break_event']>0.
        break_difference,ebreak=(difference,ed) if side=='high' else _technical_sub(latest[side],c)
        value,error=_technical_div(break_difference,denominator,ebreak,edenominator)
        out[name]=np.where(positive,0.,np.nan);radii[name]=np.where(positive,0.,np.nan)
        out[name][fired]=value[fired];radii[name][fired]=error[fired]
        name='swing_'+side+'_sequence_delta_atr';difference,ed=_technical_sub(latest[side],previous[side])
        out[name],radii[name]=_technical_div(difference,denominator,ed,edenominator)
    width,ew=_technical_sub(latest['high'],latest['low']);width=np.abs(width)
    present=(indices['high']>=0)&(indices['low']>=0)&np.isfinite(width)&(width>0.)&(indices['high']!=indices['low'])
    out['swing_impulse_present']=present.astype(np.float64)
    up=present&(indices['high']>indices['low']);down=present&(indices['low']>indices['high'])
    retracement=np.zeros(n);eretracement=np.zeros(n)
    for selected,first,second in ((up,latest['high'],c),(down,c,latest['low'])):
        difference,ed=_technical_sub(first,second)
        value,error=_technical_div(difference,np.where(selected,width,np.nan),ed,ew)
        retracement[selected]=value[selected];eretracement[selected]=error[selected]
    out['swing_retracement_from_last_impulse']=retracement;radii['swing_retracement_from_last_impulse']=eretracement
    for name in NATIVE_CORE_SWING_DISCRETE_FIELDS:
        out[name]=np.asarray(out[name],dtype=np.float64);radii[name]=np.where(np.isfinite(out[name]),0.,np.nan)
    out={name:out[name] for name in NATIVE_CORE_SWING_FIELDS};radii={name:radii[name] for name in NATIVE_CORE_SWING_FIELDS}
    return out,radii,{'source_rows':n,'lookback':2,'strict_pivots_without_tolerance':True,'gap_or_session_reset':False,
        'ATR_provided_from_shared_127_reference':True,'ATR_zero_is_unavailable':True,
        'first_valid_rows':{name:int(np.flatnonzero(np.isfinite(value))[0]) if np.isfinite(value).any() else None for name,value in out.items()},
        'positive_witnesses':{name:int(np.sum(value>0.)) for name,value in out.items()},
        'pivot_counts':{'high':int(high_pivot.sum()),'low':int(low_pivot.sum())},
        'discrete_fields':list(NATIVE_CORE_SWING_DISCRETE_FIELDS),'continuous_fields_per_TF':7}


# Integration fragment: uses the qualified127 arithmetic and root RN32 helper.
NATIVE_CORE_LOCAL_FIELDS = (
    'atr_z', '_v1_ema3_ema6_spread_atr',
    'chart.foundation_bos_up_event_age_bars', 'chart.foundation_bos_down_event_age_bars',
    'chart.foundation_choch_event_age_bars', 'smc_choch',
    'smc_sweep_up_avwap_age_bars', 'smc_sweep_up_avwap_dist_atr',
    'smc_sweep_up_avwap_dispersion_atr', 'smc_sweep_up_avwap_anchor_close_dist_atr',
    'smc_sweep_up_avwap_level_dist_atr', 'smc_sweep_up_avwap_mean_activity_ratio',
    'smc_sweep_down_avwap_age_bars', 'smc_sweep_down_avwap_dist_atr',
    'smc_sweep_down_avwap_dispersion_atr', 'smc_sweep_down_avwap_anchor_close_dist_atr',
    'smc_sweep_down_avwap_level_dist_atr', 'smc_sweep_down_avwap_mean_activity_ratio',
    'close_distance_from_ema5_bps', 'spread_bps', 'spread_bps_delta_1',
    'spread_extremes_sum_bps', 'quote_range_asymmetry_bps', 'session_change_flag',
    'mid_ema50_dist_atr', 'ema5_20_spread_atr', 'mid_ema200_dist_atr',
)
NATIVE_CORE_LOCAL_DISCRETE_FIELDS = (
    'chart.foundation_bos_up_event_age_bars', 'chart.foundation_bos_down_event_age_bars',
    'chart.foundation_choch_event_age_bars', 'smc_choch',
    'smc_sweep_up_avwap_age_bars', 'smc_sweep_down_avwap_age_bars', 'session_change_flag',
)


def _local_core_q_up(value):
    if value < 0:
        raise RuntimeError('NATIVE_CORE_LOCAL_NEGATIVE_RADIUS')
    return 0. if value == 0 else float(_technical_up(float(value)))


def _local_core_fraction_center(exact, source=None):
    from fractions import Fraction
    value = float(exact)
    radius = _local_core_q_up(abs(Fraction.from_float(value)-exact))
    if source is not None:
        radius = _technical_uadd(radius, _local_core_q_up(abs(Fraction.from_float(float(source))-exact)))
        if Fraction.from_float(value) == exact == Fraction.from_float(float(source)):
            radius = 0.
    if not np.isfinite(value) or not np.isfinite(radius):
        raise RuntimeError('NATIVE_CORE_LOCAL_OVERFLOW')
    return value, radius


def _local_core_kahan_step(state, old, new):
    # Fixed pandas2.3.3 rolling-mean add/remove compensation states. This
    # shadow conditions the error ledger only; exact rings own all values.
    s, ca, cr, same, previous = state
    if old is not None:
        y = -old-cr; t = s+y; cr = (t-s)-y; s = t
    y = new-ca; t = s+y; ca = (t-s)-y; s = t
    same = same+1 if previous == new else 1
    return (s, ca, cr, same, new)


def _local_core_atr_z(high, low, close):
    """Two exact50-rings; full-prefix Kahan and sample-Welford error shadow."""
    import math
    from fractions import Fraction
    n = len(close); out = np.full(n, np.nan); error = out.copy()
    tr = high-low
    if n > 1:
        tr[1:] = np.maximum.reduce((tr[1:], np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    if n < 99:
        return out, error, {'window':50,'ddof':1,'full_prefix':True,'shadow_used_for_values':False}
    # One shared dyadic unit exactly represents all published binary64 TRs.
    positive = tr[tr > 0.]
    exponent = math.frexp(float(positive.min()))[1]-1 if len(positive) else 0
    denominator = 1 << max(0, 53-exponent)
    ring1 = []; sum1 = 0; first = (0.,0.,0.,0,None)
    exact_ring2 = []; wire_ring2 = []; P = Q = 0
    second = (0.,0.,0.,0,None)
    count = 0; mean = ca = cr = R = 0.; Z = Fraction(0)
    variance_uniform_outputs = 0
    for row, raw in enumerate(tr):
        v = float(raw); numerator, q = v.as_integer_ratio()
        if denominator % q:
            raise RuntimeError('NATIVE_CORE_LOCAL_TR_DYADIC_DOMAIN')
        xi = numerator*(denominator//q)
        old1 = ring1.pop(0) if len(ring1) == 50 else None
        if old1 is not None:
            sum1 -= old1
        ring1.append(xi); sum1 += xi
        first = _local_core_kahan_step(first, None if old1 is None else float(Fraction(old1,denominator)), v)
        if len(ring1) < 50:
            continue
        wire = first[4] if first[3] >= 50 else max(0., first[0]/50.)
        old_exact = exact_ring2.pop(0) if len(exact_ring2) == 50 else None
        old_wire = wire_ring2.pop(0) if len(wire_ring2) == 50 else None
        if old_exact is not None:
            P -= old_exact; Q -= old_exact*old_exact
        exact_ring2.append(sum1); wire_ring2.append(wire); P += sum1; Q += sum1*sum1
        second = _local_core_kahan_step(second, old_wire, wire)
        operations = ([] if old_wire is None else [(-1, old_wire)])+[(1, wire)]
        for sign, value in operations:
            compensation = ca if sign == 1 else cr
            previous_mean = mean-compensation
            y = value-compensation; delta = y-mean; new_compensation = (delta+mean)-y
            count += sign
            if count == 0:
                mean = ca = cr = R = 0.; Z = Fraction(0)
                continue
            new_mean = mean+delta/count if sign == 1 else mean-delta/count
            A = Fraction.from_float(value)-Fraction.from_float(previous_mean)
            B = Fraction.from_float(value)-Fraction.from_float(new_mean)
            Z += sign*A*B
            if sign == 1: ca = new_compensation
            else: cr = new_compensation
            mean = new_mean
            if A == 0 or B == 0:
                local = 0.
            else:
                am = _local_core_q_up(abs(A)); bm = _local_core_q_up(abs(B))
                ra = _technical_rho(am); rb = _technical_rho(bm)
                local = _technical_uadd(_technical_umul(am,rb), _technical_umul(bm,ra),
                    _technical_umul(ra,rb), _technical_rho(_technical_umul(_technical_uadd(am,ra),_technical_uadd(bm,rb))))
            if local or R:
                total = _technical_uadd(R,local)
                R = _technical_uadd(total,_technical_rho(_technical_uadd(_local_core_q_up(abs(Z)),total)))
        if len(exact_ring2) < 50:
            continue
        # y_i=sum1/(50D). ddof1 uses49, while the mean still uses50.
        ideal_num = Fraction(50*sum1-P,2500*denominator)
        ideal_variance = Fraction(50*Q-P*P,50*49*(50*denominator)**2)
        if ideal_variance < 0:
            raise RuntimeError('NATIVE_CORE_LOCAL_SAMPLE_VARIANCE')
        # Uniform source-wire output is a domain fact, never a ledger reset.
        if second[3] >= 50:
            variance_uniform_outputs += 1
            continue
        if ideal_variance == 0:
            raise RuntimeError('NATIVE_CORE_LOCAL_ATR_Z_UNRESOLVED_ZERO_VARIANCE')
        variance, representation_error = _local_core_fraction_center(ideal_variance)
        drift = _local_core_q_up(abs(Z/49-ideal_variance))
        vr = _technical_uadd(drift,_technical_udiv(R,49.),
            _technical_rho(_technical_udiv(_technical_uadd(_local_core_q_up(abs(Z)),R),49.)),representation_error)
        lo = math.nextafter(variance-vr,-math.inf) if vr else variance
        hi = math.nextafter(variance+vr,math.inf) if vr else variance
        if lo <= 0. or not math.isfinite(hi):
            raise RuntimeError('NATIVE_CORE_LOCAL_ATR_Z_AMBIGUOUS_DENOMINATOR:row='+str(row))
        std, std_lo, std_hi = _technical_sqrt_bracket(ideal_variance.numerator,ideal_variance.denominator)
        lp,lq = lo.as_integer_ratio();hp,hq = hi.as_integer_ratio()
        _, source_lo, _ = _technical_sqrt_bracket(lp,lq)
        _, _, source_hi = _technical_sqrt_bracket(hp,hq)
        std_radius = _technical_uadd(max(abs(std-source_lo),abs(source_hi-std)),
                                    max(abs(std-std_lo),abs(std_hi-std)))
        source_mean = second[4] if second[3] >= 50 else max(0.,second[0]/50.)
        source_numerator = wire-source_mean
        numerator_value,numerator_radius = _local_core_fraction_center(ideal_num,source_numerator)
        value,radius = _technical_div(numerator_value,std,numerator_radius,std_radius)
        out[row] = float(value);error[row] = float(radius)
    out,error = _core_round32_interval(out,error)
    return out,error,{'window':50,'ddof':1,'first_possible_row':98,'full_prefix':True,
        'remove_before_add':True,'uniform_output_does_not_reset_ledger':True,
        'uniform_variance_outputs':variance_uniform_outputs,'shadow_used_for_values':False,
        'upstream_SMA50_wire_drift_included':True}


def _local_core_sweep_anchors(frame, atr, atr_radius, raw):
    import math
    from fractions import Fraction
    n = len(atr); out = {}; error = {}
    suffixes = ('age_bars','dist_atr','dispersion_atr','anchor_close_dist_atr','level_dist_atr','mean_activity_ratio')
    for side in ('up','down'):
        prefix = 'smc_sweep_'+side+'_avwap_'
        for suffix in suffixes:
            out[prefix+suffix] = np.full(n,np.nan);error[prefix+suffix] = np.full(n,np.nan)
        anchor = None
        events = raw['sweep_'+side+'_event'];levels = raw['last_'+('high' if side == 'up' else 'low')+'_price']
        for row in range(n):
            price = float(frame['close'][row]);weight = int(frame['volume'][row]);x = Fraction.from_float(price)
            if events[row] == 1.:
                level = float(levels[row])
                if not math.isfinite(level) or level <= 0.:
                    raise RuntimeError('NATIVE_CORE_LOCAL_ANCHOR_LEVEL')
                anchor = {'row':row,'close':price,'level':level,'w0':weight,'S':weight,
                    'P':weight*x,'Q':weight*x*x,'wire_S':float(weight),'wire_mean':price,'wire_M2':0.}
            elif anchor is not None:
                previous = anchor['wire_S'];total = previous+float(weight)
                delta = price-anchor['wire_mean']
                anchor['wire_mean'] = anchor['wire_mean']+(float(weight)/total)*delta
                anchor['wire_M2'] = anchor['wire_M2']+delta*delta*(previous/total)*float(weight)
                anchor['wire_S'] = total;anchor['S'] += weight;anchor['P'] += weight*x;anchor['Q'] += weight*x*x
            if anchor is None:
                continue
            age = row-anchor['row'];S = anchor['S']
            mean_exact = anchor['P']/S
            variance_exact = (S*anchor['Q']-anchor['P']*anchor['P'])/(S*S)
            if variance_exact < 0:
                raise RuntimeError('NATIVE_CORE_LOCAL_ANCHOR_VARIANCE')
            mean,mean_radius = _local_core_fraction_center(mean_exact,anchor['wire_mean'])
            std, lower, upper = _technical_sqrt_bracket(variance_exact.numerator,variance_exact.denominator)
            wire_variance = anchor['wire_M2']/anchor['wire_S']
            if not math.isfinite(wire_variance) or wire_variance < 0.:
                raise RuntimeError('NATIVE_CORE_LOCAL_ANCHOR_VARIANCE')
            wire_std = math.sqrt(wire_variance)
            std_radius = _technical_uadd(max(abs(std-lower),abs(upper-std)),
                _local_core_q_up(abs(Fraction.from_float(wire_std)-Fraction.from_float(std))))
            if variance_exact == 0 and wire_std == 0:
                std_radius = 0.
            out[prefix+'age_bars'][row] = age;error[prefix+'age_bars'][row] = 0.
            ratio_exact = Fraction(S,(age+1)*anchor['w0'])-1
            ratio_wire = anchor['wire_S']/((age+1)*float(anchor['w0']))-1.
            value,radius = _local_core_fraction_center(ratio_exact,ratio_wire)
            out[prefix+'mean_activity_ratio'][row] = value;error[prefix+'mean_activity_ratio'][row] = radius
            if np.isfinite(atr[row]):
                scale = float(atr[row]);scale_radius = float(atr_radius[row])
                differences = {
                    'dist_atr':_technical_sub(price,mean,0.,mean_radius),
                    'anchor_close_dist_atr':_technical_sub(price,anchor['close']),
                    'level_dist_atr':_technical_sub(price,anchor['level']),
                }
                for suffix,(numerator,en) in differences.items():
                    v,e = _technical_div(numerator,scale,en,scale_radius)
                    out[prefix+suffix][row] = float(v);error[prefix+suffix][row] = float(e)
                v,e = _technical_div(std,scale,std_radius,scale_radius)
                out[prefix+'dispersion_atr'][row] = float(v);error[prefix+'dispersion_atr'][row] = float(e)
        for suffix in suffixes:
            name = prefix+suffix;out[name],error[name] = _core_round32_interval(out[name],error[name])
    return out,error


def reference_native_core_local(frame, *, atr, atr_radius, raw_smc, timeframe):
    """Twenty-seven active local/scalar fields; full history, no producer call."""
    required = ('bar_start_ns','open','high','low','close','volume',
                'bid_close','ask_close','bid_high','bid_low','ask_high','ask_low')
    if any(name not in frame for name in required):
        raise RuntimeError('NATIVE_CORE_LOCAL_SOURCE_SCHEMA')
    clock = np.asarray(frame['bar_start_ns']);n = len(clock)
    delays = {'M5':5*MINUTE,'M15':15*MINUTE,'H1':60*MINUTE,'H4':240*MINUTE,'D1':DAY}
    if timeframe not in delays or clock.dtype != np.dtype('int64') or clock.ndim != 1 or not n or n >= 2**24 or np.any(clock[1:] <= clock[:-1]):
        raise RuntimeError('NATIVE_CORE_LOCAL_SOURCE_CLOCK')
    for name in required[1:]:
        a = np.asarray(frame[name]);dtype = np.dtype('int64') if name == 'volume' else np.dtype('float64')
        if a.dtype != dtype or a.shape != (n,) or not np.isfinite(a).all() or np.any(a <= 0.):
            raise RuntimeError('NATIVE_CORE_LOCAL_SOURCE_SCHEMA:'+name)
    if np.any(frame['volume'] > 2**53):
        raise RuntimeError('NATIVE_CORE_LOCAL_QUOTE_WEIGHT_PRECISION')
    h,l,c,o = (frame[name] for name in ('high','low','close','open'))
    if np.any(h < np.maximum(c,o)) or np.any(l > np.minimum(c,o)) or np.any(h < l):
        raise RuntimeError('NATIVE_CORE_LOCAL_SOURCE_GEOMETRY')
    bid,ask,bh,bl,ah,al = (frame[name] for name in ('bid_close','ask_close','bid_high','bid_low','ask_high','ask_low'))
    if np.any(ask < bid) or np.any(bh < bl) or np.any(ah < al) or np.any(ah < bh) or np.any(al < bl):
        raise RuntimeError('NATIVE_CORE_LOCAL_QUOTE_GEOMETRY')
    a,e = np.asarray(atr),np.asarray(atr_radius)
    if a.dtype != np.dtype('float64') or e.dtype != a.dtype or a.shape != (n,) or e.shape != (n,) or not np.array_equal(np.isnan(a),np.isnan(e)) or np.isinf(a).any() or np.isinf(e).any() or np.any(a[np.isfinite(a)] <= 0.) or np.any(e[np.isfinite(e)] < 0.):
        raise RuntimeError('NATIVE_CORE_LOCAL_ATR_SCHEMA')
    finite = np.flatnonzero(np.isfinite(a))
    if len(finite) and not np.isfinite(a[finite[0]:]).all():
        raise RuntimeError('NATIVE_CORE_LOCAL_ATR_SUFFIX')
    for name in ('bos_up','bos_down','choch_up','choch_down','sweep_up_event','sweep_down_event'):
        if name not in raw_smc or np.asarray(raw_smc[name]).shape != (n,) or not np.isin(raw_smc[name],(0.,1.)).all():
            raise RuntimeError('NATIVE_CORE_LOCAL_REPLAY_SCHEMA')
    for name in ('last_high_price','last_low_price'):
        if name not in raw_smc or np.asarray(raw_smc[name]).shape != (n,):
            raise RuntimeError('NATIVE_CORE_LOCAL_REPLAY_SCHEMA')
    values = {};radii = {}
    values['atr_z'],radii['atr_z'],atr_z_report = _local_core_atr_z(h,l,c)
    emas = {span:reference_technical_ema(c,span) for span in (3,5,6,20)}
    def put_ratio(name,num,den,en=0.,ed=0.,round32=False):
        v,r = _technical_div(num,den,en,ed)
        if round32:v,r = _core_round32_interval(v,r)
        values[name] = v;radii[name] = r
    gap,eg = _technical_sub(emas[3][0],emas[6][0],emas[3][1],emas[6][1])
    put_ratio('_v1_ema3_ema6_spread_atr',gap,a,eg,e)
    choch = ((raw_smc['choch_up'] > 0.)|(raw_smc['choch_down'] > 0.)).astype(np.float64)
    events = (raw_smc['bos_up'],raw_smc['bos_down'],choch)
    for name,event in zip(('chart.foundation_bos_up_event_age_bars','chart.foundation_bos_down_event_age_bars','chart.foundation_choch_event_age_bars'),events):
        rows = np.arange(n,dtype=np.int64);last = np.maximum.accumulate(np.where(event == 1.,rows,-1))
        values[name] = np.where(last >= 0,rows-last,np.nan).astype(np.float64)
        radii[name] = np.where(np.isfinite(values[name]),0.,np.nan)
    values['smc_choch'] = choch;radii['smc_choch'] = np.zeros(n)
    anchored,eb = _local_core_sweep_anchors(frame,a,e,raw_smc);values.update(anchored);radii.update(eb)
    gap,eg = _technical_sub(c,emas[5][0],0.,emas[5][1]);v,r = _technical_div(gap,c,eg,0.);v,r = _technical_scale(v,r,1e4)
    values['close_distance_from_ema5_bps'],radii['close_distance_from_ema5_bps'] = _core_round32_interval(v,r)
    gap,eg = _technical_sub(ask,bid);spread,es = _technical_div(gap,bid,eg,0.);spread,es = _technical_scale(spread,es,1e4)
    values['spread_bps'] = spread;radii['spread_bps'] = es
    previous = np.concatenate(([np.nan],spread[:-1]));ep = np.concatenate(([np.nan],es[:-1]))
    delta,ed = _technical_sub(spread,previous,es,ep)
    values['spread_bps_delta_1'],radii['spread_bps_delta_1'] = _core_round32_interval(delta,ed)
    first,ef = _technical_sub(ah,bh);second,es2 = _technical_sub(al,bl)
    total,et = _technical_sub(first,-second,ef,es2);v,r = _technical_div(total,c,et,0.);v,r = _technical_scale(v,r,1e4)
    values['spread_extremes_sum_bps'],radii['spread_extremes_sum_bps'] = _core_round32_interval(v,r)
    first,ef = _technical_sub(ah,al);second,es2 = _technical_sub(bh,bl)
    difference,ed = _technical_sub(first,second,ef,es2);v,r = _technical_div(difference,c,ed,0.);v,r = _technical_scale(v,r,1e4)
    values['quote_range_asymmetry_bps'],radii['quote_range_asymmetry_bps'] = _core_round32_interval(v,r)
    minute = ((clock+delays[timeframe]) % DAY)//MINUTE
    session = np.where((minute >= 7*60)&(minute < 12*60),1,
               np.where((minute >= 12*60)&(minute < 16*60),2,
               np.where((minute >= 16*60)&(minute < 22*60),3,0)))
    values['session_change_flag'] = np.concatenate(([0.],(session[1:] != session[:-1]).astype(np.float64)))
    radii['session_change_flag'] = np.zeros(n)
    total,et = _technical_sub(h,-l);mid,em = _technical_scale(total,et,.5)
    # The EMA helper accepts exact input observations. Its own recurrence
    # bounds are enlarged by the maximum pointwise hl2 input uncertainty.
    for span,floor,name in ((50,79,'mid_ema50_dist_atr'),(200,219,'mid_ema200_dist_atr')):
        m,er = reference_technical_ema(mid,span)
        extra = np.maximum.accumulate(em)
        valid = np.isfinite(er);er[valid] = _technical_uadd(er[valid],extra[valid])
        gap,eg = _technical_sub(mid,m,em,er);put_ratio(name,gap,a,eg,e)
        values[name][:floor] = np.nan;radii[name][:floor] = np.nan
    gap,eg = _technical_sub(emas[5][0],emas[20][0],emas[5][1],emas[20][1])
    put_ratio('ema5_20_spread_atr',gap,a,eg,e)
    values = {name:np.asarray(values[name],dtype=np.float64) for name in NATIVE_CORE_LOCAL_FIELDS}
    radii = {name:np.asarray(radii[name],dtype=np.float64) for name in NATIVE_CORE_LOCAL_FIELDS}
    first = {name:int(np.flatnonzero(np.isfinite(v))[0]) if np.isfinite(v).any() else None for name,v in values.items()}
    return values,radii,{'source_rows':n,'first_valid_rows':first,'ATR_z':atr_z_report,
        'full_history':True,'raw_SMC_before_common_mask':True,'quote_update_volume_not_GC_contract_volume':True,
        'gap_or_session_reset':False,'mid50_floor':79,'mid200_floor':219,'shadow_used_for_values':False}


NATIVE_CORE_LEVEL_FIELDS=('mtf_level_above_dist_atr', 'mtf_level_above2_dist_atr', 'mtf_level_above2_present', 'mtf_level_above_touch_count', 'mtf_level_above_recurrence_dist_atr', 'mtf_level_above_age_bars', 'mtf_level_above_bars_since_touch', 'mtf_level_above_mean_reaction_atr', 'mtf_level_above_max_reaction_atr', 'mtf_level_above_last_reaction_atr', 'mtf_level_below_dist_atr', 'mtf_level_below_present', 'mtf_level_below2_dist_atr', 'mtf_level_below2_present', 'mtf_level_below_touch_count', 'mtf_level_below_recurrence_dist_atr', 'mtf_level_below_age_bars', 'mtf_level_below_bars_since_touch', 'mtf_level_below_mean_reaction_atr', 'mtf_level_below_max_reaction_atr', 'mtf_level_below_last_reaction_atr', 'mtf_level_break_up_event', 'mtf_level_break_down_event', 'mtf_level_broken_touch_count', 'mtf_level_bars_since_break_signed', 'mtf_level_retest_hold_signed', 'mtf_level_retest_fail_signed', 'mtf_level_above_pending_retest_dist_atr', 'mtf_level_above_pending_retest_age_bars', 'mtf_level_below_pending_retest_dist_atr', 'mtf_level_below_pending_retest_age_bars', 'mtf_level_round_number_dist_50_atr', 'mtf_level_round_number_dist_100_atr')
NATIVE_CORE_LEVEL_DISCRETE_FIELDS=('mtf_level_above2_present', 'mtf_level_above_touch_count', 'mtf_level_above_age_bars', 'mtf_level_above_bars_since_touch', 'mtf_level_below_present', 'mtf_level_below2_present', 'mtf_level_below_touch_count', 'mtf_level_below_age_bars', 'mtf_level_below_bars_since_touch', 'mtf_level_break_up_event', 'mtf_level_break_down_event', 'mtf_level_broken_touch_count', 'mtf_level_bars_since_break_signed', 'mtf_level_retest_hold_signed', 'mtf_level_retest_fail_signed', 'mtf_level_above_pending_retest_age_bars', 'mtf_level_below_pending_retest_age_bars')
NATIVE_CORE_CROSS_FIELDS=('dist_to_R1_atr', 'dist_to_R2_atr', 'dist_to_S1_atr', 'dist_to_S2_atr', 'dist_to_m5_hi_atr', 'dist_to_m5_lo_atr', 'dist_to_m15_hi_atr', 'dist_to_m15_lo_atr', 'dist_to_h1_hi_atr', 'dist_to_h1_lo_atr', 'dist_to_h4_hi_atr', 'dist_to_h4_lo_atr', 'dist_to_d1_hi_atr', 'dist_to_d1_lo_atr')

NATIVE_CORE_LEVEL_EXPIRY={'M5':2721,'M15':1742,'H1':964,'H4':507,'D1':132}
NATIVE_CORE_CROSS_WINDOWS={'M5':240,'M15':192,'H1':168,'H4':168,'D1':60}
NATIVE_CORE_ROUND_GRIDS=(50.,100.)
NATIVE_CORE_SESSION_OFFSET_NS=22*60*MINUTE
NATIVE_CORE_LANE_DURATION_NS={'M5':5*MINUTE,'M15':15*MINUTE,'H1':60*MINUTE,'H4':240*MINUTE,'D1':DAY}


def _core_level_pair(a,b,ea=0.,eb=0.):
    value,error=_technical_div(a,b,ea,eb)
    return float(value),float(error)


def _core_level_difference(a,b,ea=0.,eb=0.):
    value,error=_technical_sub(a,b,ea,eb)
    return float(value),float(error)


def _core_level_validate_frame(frame,atr,atr_radius,timeframe):
    if timeframe not in NATIVE_CORE_LANE_DURATION_NS:raise RuntimeError('NATIVE_CORE_LEVEL_TIMEFRAME')
    names=('bar_start_ns','open','high','low','close','volume')
    if not set(names)<=set(frame):raise RuntimeError('NATIVE_CORE_LEVEL_SOURCE_SCHEMA')
    clock=_projection_integer_clock(frame['bar_start_ns'],'NATIVE_CORE_LEVEL_SOURCE_CLOCK')
    if not len(clock) or len(clock)>=2**24:raise RuntimeError('NATIVE_CORE_LEVEL_AGE_STORAGE')
    duration=NATIVE_CORE_LANE_DURATION_NS[timeframe];phase=NATIVE_CORE_SESSION_OFFSET_NS%duration
    if np.any((clock-phase)%duration):raise RuntimeError('NATIVE_CORE_LEVEL_SOURCE_PHASE')
    for name in names[1:]:
        v=np.asarray(frame[name]);dtype=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if v.dtype!=dtype or v.shape!=clock.shape or not np.isfinite(v).all() or np.any(v<=0.):
            raise RuntimeError('NATIVE_CORE_LEVEL_SOURCE_SCHEMA')
    h,l,c,o=(frame[n] for n in ('high','low','close','open'))
    if np.any(h<np.maximum(c,o)) or np.any(l>np.minimum(c,o)) or np.any(h<l):
        raise RuntimeError('NATIVE_CORE_LEVEL_SOURCE_GEOMETRY')
    a,e=np.asarray(atr),np.asarray(atr_radius)
    if (a.dtype!=np.dtype('float64') or e.dtype!=np.dtype('float64') or a.shape!=clock.shape or e.shape!=clock.shape
            or np.isinf(a).any() or np.isinf(e).any() or not np.array_equal(np.isnan(a),np.isnan(e))
            or np.any(a[np.isfinite(a)]<=0.) or np.any(e[np.isfinite(e)]<0.)):
        raise RuntimeError('NATIVE_CORE_LEVEL_ATR_DOMAIN')
    finite=np.flatnonzero(np.isfinite(a))
    if len(finite) and not np.isfinite(a[finite[0]:]).all():raise RuntimeError('NATIVE_CORE_LEVEL_ATR_SUFFIX')
    return clock,a,e


def reference_native_core_level(frame,*,atr,atr_radius,raw_smc,timeframe,expiry_bars):
    """Immutable ATR-admitted anchors; independently replay only serving state.

    Historical OHLC slices are read only through the current completion row.
    This replaces repeated updates of an unexposed open extreme, not any path
    population. A newborn same-bar break keeps its episode pending: only its
    eventual resolution bar is observed while broken, exactly as the owner.
    """
    from bisect import bisect_left,bisect_right,insort
    import heapq,math
    clock,a,ae=_core_level_validate_frame(frame,atr,atr_radius,timeframe);n=len(clock)
    if isinstance(expiry_bars,bool) or not isinstance(expiry_bars,int) or expiry_bars<1:
        raise RuntimeError('NATIVE_CORE_LEVEL_EXPIRY_DOMAIN')
    h,l,c=(frame[k] for k in ('high','low','close'))
    confirmations={}
    for side,key in (('high','last_high_idx'),('low','last_low_idx')):
        v=np.asarray(raw_smc[key])
        if (v.dtype!=np.dtype('int64') or v.shape!=clock.shape or np.any(v < -1)
                or np.any((v>=0)&((v<3)|(v>np.arange(n)-3)))):
            raise RuntimeError('NATIVE_CORE_LEVEL_PIVOT_CLOCK')
        confirmations[side]=v
    out={name:np.zeros(n,np.float64) for name in NATIVE_CORE_LEVEL_FIELDS}
    errors={name:np.zeros(n,np.float64) for name in out}
    own='mtf_level_bars_since_break_signed';out[own][:]=np.nan;errors[own][:]=np.nan
    chosen={name:np.full(n,np.nan) for name in ('above','above2','below','below2','pending_above','pending_below')}
    live={};active=[];origins={'high':[],'low':[]};pending=[]
    immutable={'high':[],'low':[]};deadlines=[];next_id=0;first=n;last_break=None;last_side=0
    completed=admitted=expired=held=failed=0;max_live=0;ambiguous_orders=0
    def remove(index,item):
        pos=bisect_left(index,item)
        if pos==len(index) or index[pos]!=item:raise RuntimeError('NATIVE_CORE_LEVEL_INDEX_IDENTITY')
        index.pop(pos)
    def insert_active(v):
        insort(active,(v['price'],v['id']));insort(origins[v['side']],(v['price'],v['id']))
    def remove_active(v):
        remove(active,(v['price'],v['id']));remove(origins[v['side']],(v['price'],v['id']))
    def refresh(v,t):
        v['refresh']=t;heapq.heappush(deadlines,(t+expiry_bars,v['id'],t))
    def finish(v,t):
        nonlocal completed
        window=v['window']
        if window is None or t<=window[0]:return
        start,atr0,eatr0,side0=window
        # A pending window exists only after birth==break on a tolerance tie.
        begin=t if v['status']=='pending' else start+1
        extreme=float(np.max(h[begin:t+1])) if side0=='low' else float(np.min(l[begin:t+1]))
        diff,ed=(_core_level_difference(extreme,v['price']) if side0=='low'
                 else _core_level_difference(v['price'],extreme))
        value,error=_core_level_pair(diff,atr0,ed,eatr0)
        total,et=_core_level_difference(v['sum'],-value,v['sum_error'],error)
        v['sum'],v['sum_error']=total,et;v['count']+=1;v['last'],v['last_error']=value,error
        if v['count']==1:v['max'],v['max_error']=value,error
        else:
            old,old_error=v['max'],v['max_error'];center=max(old,value)
            lower=max(math.nextafter(old-old_error,-math.inf),math.nextafter(value-error,-math.inf))
            upper=max(math.nextafter(old+old_error,math.inf),math.nextafter(value+error,math.inf))
            v['max'],v['max_error']=center,_technical_up(max(center-lower,upper-center))
        v['mean'],v['mean_error']=_core_level_pair(total,float(v['count']),et,0.)
        v['window']=None;completed+=1
    def nearest(index,t,above,want):
        # Price-group traversal retains every duplicate anchor identity, while
        # avoiding a scan of all equal-price levels on each observed bar.
        pos=bisect_right(index,(float(c[t]),next_id)) if above else bisect_right(index,(float(c[t]),next_id))-1
        candidates=[]
        while 0<=pos<len(index):
            price=index[pos][0];lo=bisect_left(index,(price,-1));hi=bisect_right(index,(price,next_id))
            delta=price-float(c[t]);distance=(delta if above else -delta)/float(a[t])
            if candidates and ae[t]>0.:
                denominator_lower=math.nextafter(float(a[t])-float(ae[t]),-math.inf)
                denominator_upper=math.nextafter(float(a[t])+float(ae[t]),math.inf)
                if denominator_lower<=0.:raise RuntimeError('NATIVE_CORE_LEVEL_SLOT_ATR_AMBIGUOUS')
                current_delta=abs(delta)
                for _,previous_id in candidates:
                    previous_delta=abs(live[previous_id]['price']-float(c[t]))
                    if current_delta==previous_delta:continue
                    numerator=math.nextafter(abs(current_delta-previous_delta),-math.inf)
                    separation=math.nextafter(numerator/denominator_upper,-math.inf)
                    magnitude=_technical_udiv(max(current_delta,previous_delta),denominator_lower)
                    if separation<=_technical_umul(2.,_technical_rho(magnitude)):
                        raise RuntimeError('NATIVE_CORE_LEVEL_SLOT_ORDER_AMBIGUOUS')
            if len(candidates)>=want:
                ranked=sorted(candidates);cut=ranked[want-1][0]
                if distance>cut:break
            ids=[entry_id for _,entry_id in index[lo:hi]
                 if index is not pending or live[entry_id]['break_at']<t]
            for entry_id in ids[:want]:candidates.append((distance,entry_id))
            pos=hi if above else lo-1
        return [live[i] for _,i in sorted(candidates)[:want]]
    def slot(v,side,t):
        if v is None:return
        prefix='mtf_level_'+side+'_';chosen[side][t]=v['price']
        for suffix,value in (('touch_count',v['touches']),('age_bars',t-v['birth']),
                             ('bars_since_touch',t-v['last_touch'])):out[prefix+suffix][t]=value
        out[prefix+'recurrence_dist_atr'][t]=v['recurrence'];errors[prefix+'recurrence_dist_atr'][t]=v['recurrence_error']
        for suffix,ekey in (('mean','mean_error'),('max','max_error'),('last','last_error')):
            out[prefix+suffix+'_reaction_atr'][t]=v[suffix];errors[prefix+suffix+'_reaction_atr'][t]=v[ekey]
        if side=='below':out[prefix+'present'][t]=1.
    for t in range(n):
        if not np.isfinite(a[t]):continue
        while deadlines and deadlines[0][0]<t:
            _,identity,at=heapq.heappop(deadlines);v=live.get(identity)
            if v is None or v['refresh']!=at:continue
            if v['status']=='active':finish(v,t);remove_active(v)
            else:remove(pending,(v['price'],identity))
            del live[identity];expired+=1
        fresh=set()
        for side,prices in (('high',h),('low',l)):
            j=t-3
            if j<3 or confirmations[side][t]!=j:continue
            price=float(prices[j]);prior=immutable[side];p=bisect_left(prior,(price,-1))
            neighbors=prior[max(0,p-1):min(len(prior),p+1)]
            if neighbors:
                candidates=[(abs(price-previous)/float(a[t]),identity,previous) for previous,identity in neighbors]
                _,_,previous=min(candidates)
                diff,ed=_core_level_difference(price,previous)
                recurrence,er=_core_level_pair(abs(diff),float(a[t]),ed,float(ae[t]))
            else:recurrence=er=0.
            v={'id':next_id,'price':price,'birth_side':side,'side':side,'birth':t,'last_touch':j,
               'touches':1,'refresh':t,'status':'active','active_since':t,'break_at':-1,'break_side':0,
               'recurrence':recurrence,'recurrence_error':er,'sum':0.,'sum_error':0.,'count':0,
               'mean':0.,'mean_error':0.,'max':0.,'max_error':0.,'last':0.,'last_error':0.,
               'window':(t,float(a[t]),float(ae[t]),side)}
            live[next_id]=v;fresh.add(next_id);insort(prior,(price,next_id));insert_active(v);refresh(v,t)
            next_id+=1;admitted+=1;first=min(first,t)
        high_break=[identity for _,identity in origins['high'][:bisect_left(origins['high'],(float(c[t]),-1))]]
        low_break=[identity for _,identity in origins['low'][bisect_right(origins['low'],(float(c[t]),next_id)):]]
        if high_break and low_break:raise RuntimeError('NATIVE_CORE_LEVEL_SAME_BAR_TWO_SIDED_BREAK')
        broken=0
        for ids,side in ((high_break,1),(low_break,-1)):
            for identity in ids:
                v=live[identity];finish(v,t);remove_active(v);v['status']='pending';v['break_at']=t;v['break_side']=side
                insort(pending,(v['price'],identity));broken=max(broken,v['touches']);last_break=t;last_side=side
        hold_sum=fail_sum=0
        left=bisect_left(pending,(float(l[t]),-1));right=bisect_right(pending,(float(h[t]),next_id))
        for _,identity in list(pending[left:right]):
            v=live[identity]
            if v['break_at']==t:continue
            finish(v,t);remove(pending,(v['price'],identity))
            if (float(c[t])-v['price'])*v['break_side']>0.:
                hold_sum+=v['break_side'];v['side']='low' if v['side']=='high' else 'high'
                v['status']='active';v['active_since']=t;insert_active(v);held+=1
            else:fail_sum+=v['break_side'];del live[identity];failed+=1
        left=bisect_left(active,(float(l[t]),-1));right=bisect_right(active,(float(h[t]),next_id))
        for _,identity in list(active[left:right]):
            if identity in fresh:continue
            v=live[identity];was=v['active_since']<t and t>0 and float(l[t-1])<=v['price']<=float(h[t-1])
            if not was:
                finish(v,t);v['touches']+=1;v['last_touch']=t;refresh(v,t)
                v['window']=(t,float(a[t]),float(ae[t]),v['side'])
        max_live=max(max_live,len(live))
        if first==n:continue
        for side,is_above in (('above',True),('below',False)):
            picks=nearest(active,t,is_above,2);slot(picks[0] if picks else None,side,t)
            if len(picks)>1:chosen[side+'2'][t]=picks[1]['price'];out['mtf_level_'+side+'2_present'][t]=1.
            waiting=nearest(pending,t,is_above,1)
            if waiting:
                v=waiting[0];chosen['pending_'+side][t]=v['price']
                out['mtf_level_'+side+'_pending_retest_age_bars'][t]=t-v['break_at']
        out['mtf_level_break_up_event'][t]=bool(high_break);out['mtf_level_break_down_event'][t]=bool(low_break)
        out['mtf_level_broken_touch_count'][t]=broken
        if last_break is not None:out[own][t]=last_side*(t-last_break);errors[own][t]=0.
        out['mtf_level_retest_hold_signed'][t]=int(hold_sum>0)-int(hold_sum<0)
        out['mtf_level_retest_fail_signed'][t]=int(fail_sum>0)-int(fail_sum<0)
    for side in ('above','below'):
        for key,name in ((side,'mtf_level_'+side+'_dist_atr'),(side+'2','mtf_level_'+side+'2_dist_atr'),
                         ('pending_'+side,'mtf_level_'+side+'_pending_retest_dist_atr')):
            selected=chosen[key];present=np.isfinite(selected)
            diff,ed=_technical_sub(selected,c) if side=='above' else _technical_sub(c,selected)
            value,error=_technical_div(diff,a,ed,ae);out[name]=np.where(present,value,0.);errors[name]=np.where(present,error,0.)
    for grid in NATIVE_CORE_ROUND_GRIDS:
        # Canonical RN64 division/addition precedes floor, including half ties.
        center=grid*np.floor(c/grid+.5)
        diff,ed=_technical_sub(c,center);value,error=_technical_div(diff,a,ed,ae)
        out['mtf_level_round_number_dist_'+str(int(grid))+'_atr']=value
        errors['mtf_level_round_number_dist_'+str(int(grid))+'_atr']=error
    for name in out:
        out[name][:first]=np.nan;errors[name][:first]=np.nan
    return out,errors,{'source_rows':n,'first_valid_row':None if first==n else first,'timeframe':timeframe,
        'expiry_bars':expiry_bars,'pivots_before_ATR_are_not_admitted':True,'new_immutable_anchors':admitted,
        'completed_reactions':completed,'eligibility_expiries':expired,'held_retests':held,'failed_retests':failed,
        'maximum_live_eligible_records':max_live,'ambiguous_slot_orders':ambiguous_orders,
        'reaction_queries_end_at_current_row':True,'same_bar_birth_break_pending_episode_preserved':True,
        'gap_or_session_reset':False,'recurrence_threshold_affects_outputs':False}


def _core_cross_reference_axes(raw_m5):
    names=('bar_start_ns','open','high','low','close','volume')
    clock=np.asarray(raw_m5['bar_start_ns']);t=_projection_integer_clock(clock,'NATIVE_CORE_CROSS_M5_CLOCK',grid=5*MINUTE)
    if not len(t):raise RuntimeError('NATIVE_CORE_CROSS_M5_EMPTY')
    for name in names[1:]:
        v=np.asarray(raw_m5[name]);dtype=np.dtype('int64') if name=='volume' else np.dtype('float64')
        if v.dtype!=dtype or v.shape!=t.shape or not np.isfinite(v).all() or np.any(v<=0.):raise RuntimeError('NATIVE_CORE_CROSS_M5_DOMAIN')
    if (np.any(raw_m5['high']<np.maximum(raw_m5['open'],raw_m5['close']))
            or np.any(raw_m5['low']>np.minimum(raw_m5['open'],raw_m5['close']))
            or np.any(raw_m5['high']<raw_m5['low'])):raise RuntimeError('NATIVE_CORE_CROSS_M5_GEOMETRY')
    axes={}
    for tf,duration in NATIVE_CORE_LANE_DURATION_NS.items():
        phase=NATIVE_CORE_SESSION_OFFSET_NS%duration;labels=((t-phase)//duration)*duration+phase
        stamps,first,count=np.unique(labels,return_index=True,return_counts=True)
        # Unlike the feature-lane axis, the raw context owner retains both
        # leading and trailing observed partial bins. As-of selection censors
        # reference bins whose declared close lies after a query's availability.
        axes[tf]={'bar_start_ns':stamps,'high':np.maximum.reduceat(raw_m5['high'],first),
            'low':np.minimum.reduceat(raw_m5['low'],first),'close':raw_m5['close'][first+count-1]}
    return axes


def reference_native_core_cross(frame,*,atr,atr_radius,raw_m5,timeframe):
    """Four strict-prior-session pivots plus ten trailing closed-reference extrema."""
    from bisect import bisect_left,bisect_right,insort
    clock,a,ae=_core_level_validate_frame(frame,atr,atr_radius,timeframe);n=len(clock);c=frame['close']
    axes=_core_cross_reference_axes(raw_m5);duration=NATIVE_CORE_LANE_DURATION_NS[timeframe];availability=clock+duration
    # The source's intermediate float32 ATR-bps wire is a semantic operation.
    bps,ebps=_technical_scale(*_technical_div(a,c,ae,0.),1e4)
    bps,ebps=_core_round32_interval(bps,ebps)
    scaled,es=_technical_div(bps,np.full(n,1e4),ebps,0.)
    cross_atr=scaled*c;propagated=_technical_umul(es,np.abs(c));finite=np.isfinite(cross_atr)
    if np.any(cross_atr[finite]<=0.):raise RuntimeError('NATIVE_CORE_CROSS_ATR_ROUNDTRIP_DOMAIN')
    ecross=np.full(n,np.nan)
    ecross[finite]=_technical_uadd(propagated[finite],
        _technical_rho(_technical_uadd(np.abs(cross_atr[finite]),propagated[finite])),_technical_rho(cross_atr[finite]))
    out={name:np.full(n,np.nan) for name in NATIVE_CORE_CROSS_FIELDS};radii={name:v.copy() for name,v in out.items()}
    daily=axes['D1'];H,L,C=(daily[k] for k in ('high','low','close'))
    total,et=_technical_sub(H,-L);total,et=_technical_sub(total,-C,et,0.)
    pp,epp=_technical_div(total,np.full(len(H),3.),et,0.);twopp,etwopp=_technical_scale(pp,epp,2.)
    ran,er=_technical_sub(H,L);levels={}
    levels['R1']=_technical_sub(twopp,L,etwopp,0.);levels['R2']=_technical_sub(pp,-ran,epp,er)
    levels['S1']=_technical_sub(twopp,H,etwopp,0.);levels['S2']=_technical_sub(pp,ran,epp,er)
    session=((availability-NATIVE_CORE_SESSION_OFFSET_NS)//DAY)*DAY+NATIVE_CORE_SESSION_OFFSET_NS
    prior=np.searchsorted(daily['bar_start_ns'],session,side='left')-1;observed=prior>=0
    for name,(value,error) in levels.items():
        selected=np.full(n,np.nan);se=selected.copy();selected[observed]=value[prior[observed]];se[observed]=error[prior[observed]]
        diff,ed=_technical_sub(c,selected,0.,se);out['dist_to_'+name+'_atr'],radii['dist_to_'+name+'_atr']=_technical_div(diff,cross_atr,ed,ecross)
    extrema={};full=np.isfinite(cross_atr)
    for tf,lookback in NATIVE_CORE_CROSS_WINDOWS.items():
        axis=axes[tf];right=np.searchsorted(axis['bar_start_ns'],availability-NATIVE_CORE_LANE_DURATION_NS[tf],side='right')
        complete=right>=lookback;full &= complete
        upper=np.full(n,np.nan);lower=upper.copy();highs=[];lows=[];cursor=0
        for row,r in enumerate(right):
            r=int(r)
            while cursor<r:
                insort(highs,float(axis['high'][cursor]));insort(lows,float(axis['low'][cursor]));cursor+=1
                if cursor>lookback:
                    old=cursor-lookback-1
                    highs.pop(bisect_left(highs,float(axis['high'][old])));lows.pop(bisect_left(lows,float(axis['low'][old])))
            if not complete[row]:continue
            pos=bisect_right(highs,float(c[row]));upper[row]=highs[pos] if pos<len(highs) else highs[-1]
            pos=bisect_left(lows,float(c[row]));lower[row]=lows[pos-1] if pos else lows[0]
        extrema[tf]=(upper,lower)
    for tf,(upper,lower) in extrema.items():
        for suffix,first,second in (('hi',upper,c),('lo',c,lower)):
            diff,ed=_technical_sub(first,second);value,error=_technical_div(diff,cross_atr,ed,ecross)
            name='dist_to_'+tf.lower()+'_'+suffix+'_atr';out[name]=np.where(full,value,np.nan);radii[name]=np.where(full,error,np.nan)
    first={name:int(np.flatnonzero(np.isfinite(v))[0]) if np.isfinite(v).any() else None for name,v in out.items()}
    return out,radii,{'source_rows':n,'timeframe':timeframe,'raw_M5_rows':len(raw_m5['bar_start_ns']),
        'first_valid_rows':first,'ATR_uses_explicit_RN32_bps_roundtrip':True,'hidden_cache_inputs':False,
        'reference_rows':{tf:len(v['bar_start_ns']) for tf,v in axes.items()},
        'reference_first_labels_ns':{tf:int(v['bar_start_ns'][0]) for tf,v in axes.items()},
        'leading_partial_reference_bins_retained':True,'reference_cutoff':'lane_start+lane_duration-reference_duration',
        'strict_prior_observed_D1_session':True,'session_offset_hours_UTC':22,'all_ten_extrema_share_full_window_mask':True}


NATIVE_CORE_NUMERIC_PLAN_ID='NATIVE_M5_REMAINING_ACTIVE_CORE_NUMERIC_REFERENCE_20261021_130'
NATIVE_CORE_NUMERIC_POLICY_KEY='remaining_active_core_numeric_reference_130'
NATIVE_CORE_NUMERIC_FIELDS=('swing_dist_last_swing_high_atr', 'swing_dist_last_swing_low_atr', 'swing_bars_since_swing_high', 'swing_bars_since_swing_low', 'swing_retracement_from_last_impulse', 'swing_impulse_present', 'swing_high_break_event', 'swing_low_break_event', 'swing_high_break_displacement_atr', 'swing_low_break_displacement_atr', 'bars_since_swing_high_break', 'bars_since_swing_low_break', 'swing_high_sequence_delta_atr', 'swing_low_sequence_delta_atr', 'consecutive_higher_lows_count', 'consecutive_lower_highs_count', 'consecutive_higher_highs_count', 'consecutive_lower_lows_count', 'swing_high_level_intact', 'swing_low_level_intact', 'mtf_level_above_dist_atr', 'mtf_level_above2_dist_atr', 'mtf_level_above2_present', 'mtf_level_above_touch_count', 'mtf_level_above_recurrence_dist_atr', 'mtf_level_above_age_bars', 'mtf_level_above_bars_since_touch', 'mtf_level_above_mean_reaction_atr', 'mtf_level_above_max_reaction_atr', 'mtf_level_above_last_reaction_atr', 'mtf_level_below_dist_atr', 'mtf_level_below_present', 'mtf_level_below2_dist_atr', 'mtf_level_below2_present', 'mtf_level_below_touch_count', 'mtf_level_below_recurrence_dist_atr', 'mtf_level_below_age_bars', 'mtf_level_below_bars_since_touch', 'mtf_level_below_mean_reaction_atr', 'mtf_level_below_max_reaction_atr', 'mtf_level_below_last_reaction_atr', 'mtf_level_break_up_event', 'mtf_level_break_down_event', 'mtf_level_broken_touch_count', 'mtf_level_bars_since_break_signed', 'mtf_level_retest_hold_signed', 'mtf_level_retest_fail_signed', 'mtf_level_above_pending_retest_dist_atr', 'mtf_level_above_pending_retest_age_bars', 'mtf_level_below_pending_retest_dist_atr', 'mtf_level_below_pending_retest_age_bars', 'mtf_level_round_number_dist_50_atr', 'mtf_level_round_number_dist_100_atr', 'atr_z', '_v1_ema3_ema6_spread_atr', 'chart.foundation_bos_up_event_age_bars', 'chart.foundation_bos_down_event_age_bars', 'chart.foundation_choch_event_age_bars', 'smc_choch', 'smc_sweep_up_avwap_age_bars', 'smc_sweep_up_avwap_dist_atr', 'smc_sweep_up_avwap_dispersion_atr', 'smc_sweep_up_avwap_anchor_close_dist_atr', 'smc_sweep_up_avwap_level_dist_atr', 'smc_sweep_up_avwap_mean_activity_ratio', 'smc_sweep_down_avwap_age_bars', 'smc_sweep_down_avwap_dist_atr', 'smc_sweep_down_avwap_dispersion_atr', 'smc_sweep_down_avwap_anchor_close_dist_atr', 'smc_sweep_down_avwap_level_dist_atr', 'smc_sweep_down_avwap_mean_activity_ratio', 'close_distance_from_ema5_bps', 'spread_bps', 'spread_bps_delta_1', 'spread_extremes_sum_bps', 'quote_range_asymmetry_bps', 'session_change_flag', 'mid_ema50_dist_atr', 'ema5_20_spread_atr', 'mid_ema200_dist_atr', 'dist_to_R1_atr', 'dist_to_R2_atr', 'dist_to_S1_atr', 'dist_to_S2_atr', 'dist_to_m5_hi_atr', 'dist_to_m5_lo_atr', 'dist_to_m15_hi_atr', 'dist_to_m15_lo_atr', 'dist_to_h1_hi_atr', 'dist_to_h1_lo_atr', 'dist_to_h4_hi_atr', 'dist_to_h4_lo_atr', 'dist_to_d1_hi_atr', 'dist_to_d1_lo_atr')
NATIVE_CORE_NUMERIC_COLUMNS=(50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 190, 196, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 223, 224, 225, 226, 227, 229, 233, 235, 237, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252)

NATIVE_CORE_QUOTE_FIELDS=('bid_close','ask_close','bid_high','bid_low','ask_high','ask_low')


def _core_round32_interval(values,radii):
    """Propagate an internal RN32 wire without reading the target cache."""
    value,radius=np.broadcast_arrays(np.asarray(values,dtype=np.float64),np.asarray(radii,dtype=np.float64))
    valid=np.isfinite(value)
    if (np.any(np.isinf(value)) or np.any(valid&(~np.isfinite(radius)|(radius<0.)))):
        raise RuntimeError('NATIVE_CORE_NUMERIC_F32_INTERVAL_DOMAIN')
    out=np.full(value.shape,np.nan,dtype=np.float64);bound=out.copy()
    if not np.any(valid):return out,bound
    v,r=value[valid],radius[valid]
    # Zero radius means the binary64 input wire is exact. Otherwise both
    # subtraction/addition endpoints are expanded before the monotone cast.
    lo=np.where(r==0.,v,np.nextafter(v-r,-np.inf))
    hi=np.where(r==0.,v,np.nextafter(v+r,np.inf))
    with np.errstate(over='ignore',invalid='ignore'):
        center=v.astype(np.float32).astype(np.float64)
        lower=lo.astype(np.float32).astype(np.float64)
        upper=hi.astype(np.float32).astype(np.float64)
    if not (np.isfinite(center).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise RuntimeError('NATIVE_CORE_NUMERIC_F32_INTERVAL_OVERFLOW')
    error=np.maximum(center-lower,upper-center)
    # Differences of F32 endpoints are enclosed by one outward F64 step.
    bound[valid]=np.where(error==0.,0.,np.nextafter(error,np.inf));out[valid]=center
    return out,bound


def _core_native_quote_footer(path,seal):
    rows=_technical_native_footer(path,seal)
    schema=pq.ParquetFile(path).schema_arrow
    if any(name not in schema.names or schema.field(name).type!=pa.float64() for name in NATIVE_CORE_QUOTE_FIELDS):
        raise RuntimeError('NATIVE_CORE_NUMERIC_QUOTE_SCHEMA')
    return rows


def native_core_resample_frame(m5,timeframe):
    """Use the qualified native axis; aggregate each genuine quote side."""
    base=('bar_start_ns','open','high','low','close','volume')
    if set(m5)!=set(base+NATIVE_CORE_QUOTE_FIELDS):
        raise RuntimeError('NATIVE_CORE_NUMERIC_QUOTE_SOURCE_SCHEMA')
    result=native_resample_calendar_frame({name:m5[name] for name in base},timeframe)
    t=m5['bar_start_ns']
    for name in NATIVE_CORE_QUOTE_FIELDS:
        a=np.asarray(m5[name])
        if a.dtype!=np.dtype('float64') or a.shape!=t.shape or not np.isfinite(a).all() or np.any(a<=0.):
            raise RuntimeError('NATIVE_CORE_NUMERIC_QUOTE_SOURCE_SCHEMA')
    for side in ('bid','ask'):
        if (np.any(m5[side+'_high']<m5[side+'_low']) or np.any(m5[side+'_high']<m5[side+'_close'])
                or np.any(m5[side+'_low']>m5[side+'_close'])):
            raise RuntimeError('NATIVE_CORE_NUMERIC_QUOTE_GEOMETRY')
    if any(np.any(m5['ask_'+part]<m5['bid_'+part]) for part in ('close','high','low')):
        raise RuntimeError('NATIVE_CORE_NUMERIC_CROSSED_QUOTE')
    durations={'M5':5*MINUTE,'M15':15*MINUTE,'H1':60*MINUTE,'H4':240*MINUTE,'D1':DAY}
    duration=durations[timeframe];phase=(22*60*MINUTE)%duration
    labels=((t-phase)//duration)*duration+phase
    stamps,first,count=np.unique(labels,return_index=True,return_counts=True)
    pos=np.searchsorted(stamps,result['bar_start_ns'])
    if not np.array_equal(stamps[pos],result['bar_start_ns']):
        raise RuntimeError('NATIVE_CORE_NUMERIC_QUOTE_AXIS')
    for name in NATIVE_CORE_QUOTE_FIELDS:
        values=m5[name]
        aggregated=(np.maximum.reduceat(values,first) if name.endswith('_high') else
            np.minimum.reduceat(values,first) if name.endswith('_low') else values[first+count-1])
        result[name]=aggregated[pos]
    return result


def _core_reference_atr(frame):
    high,low,close=(frame[name] for name in ('high','low','close'))
    tr=high-low
    if len(tr)>1:tr[1:]=np.maximum.reduce((tr[1:],np.abs(high[1:]-close[:-1]),np.abs(low[1:]-close[:-1])))
    return reference_technical_wilder(tr,14)


def _core_compare_owner_blocks(m5,timeframe,clock,cache_path,expiry_bars):
    """Compare one owner at a time; never merge 94 full-history surfaces."""
    from time import monotonic
    raw=native_core_resample_frame(m5,timeframe)
    positions=np.searchsorted(raw['bar_start_ns'],clock)
    if (np.any(positions>=len(raw['bar_start_ns'])) or not np.array_equal(raw['bar_start_ns'][positions],clock)
            or not np.array_equal(raw['bar_start_ns'][(raw['bar_start_ns']>=clock[0])&(raw['bar_start_ns']<=clock[-1])],clock)):
        raise RuntimeError('NATIVE_CORE_NUMERIC_EXACT_CACHE_TIMESTAMP_MAPPING:'+timeframe)
    atr,atr_radius=_core_reference_atr(raw)
    raw_smc=reference_native_smc_replay(raw['high'],raw['low'],raw['close'])
    for vector in (*raw.values(),*m5.values(),atr,atr_radius,*raw_smc.values()):vector.setflags(write=False)
    matrix=np.load(cache_path,mmap_mode='r',allow_pickle=False)
    column_by_name=dict(zip(NATIVE_CORE_NUMERIC_FIELDS,NATIVE_CORE_NUMERIC_COLUMNS));reports={};helpers={}
    blocks=(('swing20',NATIVE_CORE_SWING_FIELDS,NATIVE_CORE_SWING_DISCRETE_FIELDS,
        lambda:reference_native_core_swing(raw,atr=atr,atr_radius=atr_radius)),
        ('local27',NATIVE_CORE_LOCAL_FIELDS,NATIVE_CORE_LOCAL_DISCRETE_FIELDS,
        lambda:reference_native_core_local(raw,atr=atr,atr_radius=atr_radius,raw_smc=raw_smc,timeframe=timeframe)),
        ('level33',NATIVE_CORE_LEVEL_FIELDS,NATIVE_CORE_LEVEL_DISCRETE_FIELDS,
        lambda:reference_native_core_level(raw,atr=atr,atr_radius=atr_radius,raw_smc=raw_smc,timeframe=timeframe,expiry_bars=expiry_bars)),
        ('cross14',NATIVE_CORE_CROSS_FIELDS,(),
        lambda:reference_native_core_cross(raw,atr=atr,atr_radius=atr_radius,raw_m5=m5,timeframe=timeframe)))
    for owner,names,discrete,compute in blocks:
        started=monotonic();reference,bounds,helper=compute()
        if (set(reference)!=set(names) or set(bounds)!=set(names) or any(name in reports for name in names)
                or any(np.shape(reference[name])!=np.shape(raw['bar_start_ns']) or np.shape(bounds[name])!=np.shape(raw['bar_start_ns']) for name in names)):
            raise RuntimeError('NATIVE_CORE_NUMERIC_OWNER_SURFACE:'+owner)
        ref={name:reference[name][positions] for name in names};radii={name:bounds[name][positions] for name in names}
        target={name:np.asarray(matrix[:,column_by_name[name]]) for name in names}
        report=compare_technical_reference(ref,target,radii,discrete_fields=discrete)
        for name,stat in report.items():
            if not stat['numeric_reference_qualified']:
                raise RuntimeError('NATIVE_CORE_NUMERIC_ACTIVE_FIELD_UNOBSERVED:'+timeframe+':'+name)
            stat['compact_profile_active']=True
            stat['positive_reference_values']=int(np.sum(ref[name]>0.));stat['zero_reference_values']=int(np.sum(ref[name]==0.))
        reports.update(report);helpers[owner]=dict(helper,oracle_and_cache_compare_elapsed_seconds=monotonic()-started)
        del reference,bounds,ref,radii,target,report
    if set(reports)!=set(NATIVE_CORE_NUMERIC_FIELDS):raise RuntimeError('NATIVE_CORE_NUMERIC_OWNER_UNION')
    return reports,helpers,len(clock)


def run_technical_numeric(binding_path,*,trend_events=False,native_smc=False,remaining_core=False):
    if sum(bool(v) for v in (trend_events,native_smc,remaining_core))>1:raise RuntimeError('TECHNICAL_NUMERIC_MUTUALLY_EXCLUSIVE_MODES')
    prefix='NATIVE_CORE_NUMERIC' if remaining_core else 'NATIVE_SMC_NUMERIC' if native_smc else 'TREND_EVENT_NUMERIC' if trend_events else 'TECHNICAL_NUMERIC'
    plan_id=NATIVE_CORE_NUMERIC_PLAN_ID if remaining_core else NATIVE_SMC_NUMERIC_PLAN_ID if native_smc else TREND_EVENT_NUMERIC_PLAN_ID if trend_events else TECHNICAL_NUMERIC_PLAN_ID
    policy_key=NATIVE_CORE_NUMERIC_POLICY_KEY if remaining_core else NATIVE_SMC_NUMERIC_POLICY_KEY if native_smc else TREND_EVENT_NUMERIC_POLICY_KEY if trend_events else TECHNICAL_NUMERIC_POLICY_KEY
    schema='gx1_native_core_numeric_audit_plan_v1' if remaining_core else 'gx1_native_smc_numeric_audit_plan_v1' if native_smc else 'gx1_trend_event_numeric_audit_plan_v1' if trend_events else 'gx1_technical_numeric_audit_plan_v1'
    fields=NATIVE_CORE_NUMERIC_FIELDS if remaining_core else NATIVE_SMC_NUMERIC_FIELDS if native_smc else TREND_EVENT_NUMERIC_FIELDS if trend_events else TECHNICAL_NUMERIC_FIELDS
    columns=NATIVE_CORE_NUMERIC_COLUMNS if remaining_core else tuple(range(70,94)) if native_smc else tuple(range(94,111)) if trend_events else tuple(range(23))
    active_count=94 if remaining_core else 17 if (trend_events or native_smc) else 11
    root=Path('/home/andre2/GX1_RUNS')/plan_id
    if binding_path!=root/(prefix+'_EXECUTION_BINDING.json'):raise RuntimeError(prefix+'_BINDING_PATH')
    io.require_regular_file(binding_path,prefix+'_BINDING');binding=io.read_json(binding_path);plan_path=root/'PLAN.json'
    io.verify_digest(plan_path,binding['plan_sha256'],prefix+'_PLAN');plan=io.read_json(plan_path)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],prefix+'_POLICY')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work'][policy_key]
    if (plan['schema_version']!=schema or plan['plan_id']!=plan_id
            or plan['run_root']!=str(root) or plan['policy_key']!=policy_key or plan['max_invocations']!=1
            or gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0
            or gate['execution_authorized'] is not False or gate['relaunch_allowed'] is not False
            or gate['plan']!={'path':str(plan_path),'sha256':binding['plan_sha256']}
            or policy['training_enabled'] is not False or io.git_head()!=binding['source_head']
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):
        raise RuntimeError(prefix+'_AUTHORITY')
    if set(binding['code_sha256'])!=set(plan['code_paths']):raise RuntimeError(prefix+'_CODE_INVENTORY')
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,prefix+'_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    io.write_json(root/(prefix+'_CLAIM.json'),{'invocation':1,'source_head':binding['source_head'],'plan_sha256':binding['plan_sha256']})
    staging=root/('.'+prefix+'_RESULTS.staging');staging.mkdir(exist_ok=False)
    metadata_sha={};identities={};data_files={};runtime_files={}
    identity=lambda st:(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    def verified(ref,label):
        path=Path(ref['path']);io.require_regular_file(path,label)
        if not path.is_absolute() or path!=path.resolve():raise RuntimeError(prefix+'_CANONICAL_PATH')
        before=identity(path.stat());io.verify_digest(path,ref['sha256'],label)
        if identity(path.stat())!=before:raise RuntimeError(prefix+'_INPUT_CHANGED')
        identities[str(path)]=before;return path
    def metadata(ref):
        path=verified(ref,prefix+'_METADATA');metadata_sha[str(path)]=ref['sha256'];return io.read_json(path)
    design=metadata(plan['design']);spec=metadata(plan['error_spec'])
    if (design['run_id']!=plan_id or spec['design']!=plan['design']
            or spec['frozen_before_any_cached_numeric_comparison'] is not True or design['references']!=plan['references']
            or tuple(design['audited_fields'])!=fields):raise RuntimeError(prefix+'_FROZEN_ERROR_SPEC')
    if trend_events or native_smc or remaining_core:
        inherited=metadata(spec['inherited_127_arithmetic_spec'])
        if inherited['frozen_before_any_cached_numeric_comparison'] is not True:raise RuntimeError(prefix+'_INHERITED_ERROR_SPEC')
    if trend_events:
        clarification=metadata(gate['design_cache_column_clarification'])
        if (inherited['frozen_before_any_cached_numeric_comparison'] is not True or clarification['design']!=plan['design']
                or tuple(clarification['cache_column_indices'])!=columns or tuple(clarification['audited_fields'])!=fields
                or clarification['bound_whole_data_files']!=28 or clarification['bound_whole_payload_bytes']!=design['known_payload_bytes']['data_total']
                or clarification['logical_17selected_feature_bytes']!=sum(design['expected_cache_rows'].values())*17*4):
            raise RuntimeError(prefix+'_INHERITED_SPEC_OR_COLUMN_CLARIFICATION')
    seal_stamp=pd.Timestamp(plan['TEST_seal_utc'])
    from gx1.contracts.gc_source_disposition_v1 import WINDOW_END_NS
    if seal_stamp.tzinfo is None or seal_stamp.value!=WINDOW_END_NS:raise RuntimeError(prefix+'_TEST_SEAL')
    seal=seal_stamp.value;refs=plan['references'];cache=metadata(refs['cache_manifest']);prebuilt=metadata(refs['prebuilt_manifest'])
    native=metadata(refs['native_m5_manifest']);profile=metadata(refs['compact_profile_proposal']);previous=metadata(refs['previous_native_context_review'])
    if trend_events or native_smc or remaining_core:
        prior=metadata(refs['previous_technical_numeric_review'])
        if (prior['status']!='PASS_SCOPED_115_TECHNICAL_OUTPUTS_ALL5TF' or prior['source_unchanged'] is not True
                or prior['test_data_used'] is not False or prior['qualified_outputs']!=115
                or prior['error_spec']!=spec['inherited_127_arithmetic_spec']):
            raise RuntimeError(prefix+'_PRIMITIVE_REUSE')
        if tuple(design['cache_column_indices'])!=columns or tuple(cache['feature_names'][j] for j in columns)!=fields:
            raise RuntimeError(prefix+'_CACHE_FIELD_ORDER')
    if native_smc:
        trend=metadata(refs['previous_trend_numeric_review']);index=metadata(refs['qualified_mtf_numeric_index'])
        if (trend['status']!='PASS_SCOPED_85_EXACT_TREND_OUTPUTS_ALL5TF' or trend['source_unchanged'] is not True
                or trend['test_data_used'] is not False or trend['qualified_outputs']!=85
                or index['cache_manifest']!=refs['cache_manifest'] or index['qualified_output_records']!=200
                or len(index['rows'])!=200 or any(row['name'] in fields for row in index['rows'])
                or tuple(design['discrete_fields'])!=NATIVE_SMC_DISCRETE_FIELDS
                or design['known_payload_bytes']['logical_24selected_feature_bytes']!=sum(design['expected_cache_rows'].values())*24*4):
            raise RuntimeError(prefix+'_PRIOR_INDEX_OR_FIELD_CONTRACT')
    if remaining_core:
        if set(spec['arithmetic_parts'])!={'reader','swing','local','level_cross'}:raise RuntimeError(prefix+'_ARITHMETIC_PART_INVENTORY')
        arithmetic_parts={name:metadata(ref) for name,ref in spec['arithmetic_parts'].items()}
        smc=metadata(refs['previous_native_smc_numeric_review']);index=metadata(refs['qualified_mtf_numeric_index'])
        constants=cache.get('v29_registry_constants',{})
        expected_expiry={'M5':2721,'M15':1742,'H1':964,'H4':507,'D1':132}
        discrete=set(NATIVE_CORE_SWING_DISCRETE_FIELDS+NATIVE_CORE_LOCAL_DISCRETE_FIELDS+NATIVE_CORE_LEVEL_DISCRETE_FIELDS)
        if (smc['status']!='PASS_SCOPED_120_NATIVE_SMC_GEOMETRY_OUTPUTS_ALL5TF' or smc['source_unchanged'] is not True
                or smc['test_data_used'] is not False or smc['qualified_outputs']!=120
                or index['cache_manifest']!=refs['cache_manifest'] or index['qualified_output_records']!=320
                or len(index['rows'])!=320 or any(row['name'] in fields for row in index['rows'])
                or set(design['discrete_fields'])!=discrete or len(discrete)!=37
                or constants.get('schema_version')!='htf_v4_v29_registry_constants_v9'
                or constants.get('contract_sha256')!='a222c3c1ff8d22683d209ef6cc9e7827bdbb5a374518ba1dfe914c7e9121e6a2'
                or constants.get('level_expiry_bars')!=expected_expiry or design['level_expiry_bars']!=expected_expiry
                or design['known_payload_bytes']['logical_94selected_feature_bytes']!=sum(design['expected_cache_rows'].values())*94*4):
            raise RuntimeError(prefix+'_PRIOR_INDEX_OR_CORE_CONTRACT')
    runtime=design['runtime_contract']
    for key in ('metadata','window_kernel'):
        path=verified(runtime[key],prefix+'_RUNTIME');runtime_files[str(path)]={'sha256':runtime[key]['sha256'],'bytes':path.stat().st_size}
    from pandas._libs.window import aggregations as runtime_aggregations
    actual_package=Path(pd.__file__).resolve().parent;expected_kernel=actual_package/'_libs/window/aggregations.cpython-310-x86_64-linux-gnu.so'
    if (pd.__version__!=runtime['pandas_version'] or expected_kernel!=Path(runtime['window_kernel']['path'])
            or Path(runtime_aggregations.__file__).resolve()!=expected_kernel):
        raise RuntimeError(prefix+'_RUNTIME_MODULE')
    cache_root=Path(refs['cache_manifest']['path']).parent;source=native['out_root'];lineage=prebuilt['native_m5_source']
    if (cache['m5_prebuilt_source']!=prebuilt['output_parquet'] or cache['m5_prebuilt_source_sha256']!=prebuilt['output_parquet_sha256']
            or cache['feature_count']!=253 or cache['feature_names'][:23]!=list(TECHNICAL_NUMERIC_FIELDS)
            or prebuilt['decision']!='PASS' or prebuilt['timeframe']!='M5'
            or {k:lineage[k] for k in ('manifest_path','manifest_sha256')}!=
                {'manifest_path':refs['native_m5_manifest']['path'],'manifest_sha256':refs['native_m5_manifest']['sha256']}
            or lineage['root']!=source or native['schema_version']!='xau_canonical_native_source_v3'
            or native['instrument']!='XAU_USD' or native['timeframe']!='M5' or native['prices']!='MBA'
            or native['completion_field']!='complete' or native['completion_value'] is not True
            or native['bar_duration_seconds']!=300 or native['decision_available_offset_seconds']!=300
            or native['timestamp_semantics']!='bar_start_utc' or pd.Timestamp(native['requested_end_utc_exclusive']).value!=seal
            or previous['source_unchanged'] is not True or previous['test_data_used'] is not False
            or previous['independent_stored_M5_availability_projection_proven'] is not True):
        raise RuntimeError(prefix+'_SOURCE_LINEAGE')
    parts=lineage['part_paths'];years=['year='+str(year) for year in range(2009,2027)]
    if len(parts)!=18 or set(native['year_sha256'])!=set(years) or set(native['year_rows'])!=set(years):raise RuntimeError(prefix+'_YEAR_INVENTORY')
    for year,path in zip(years,parts):
        if path!=str(Path(source)/year/'part-000.parquet') or lineage['part_sha256'][path]!=native['year_sha256'][year]:
            raise RuntimeError(prefix+'_YEAR_BINDING')
        data_files[path]={'sha256':native['year_sha256'][year],'kind':'native','year':year}
    for tf in design['timeframes']:
        info=cache['tfs'][tf]
        if info['feature_count']!=253 or info['n_bars']!=design['expected_cache_rows'][tf]:raise RuntimeError(prefix+'_CACHE_DIMENSION')
        for kind,name,sha,bytekey in (('timestamps','ts_npy','ts_npy_sha256','ts_npy_size_bytes'),
                ('features','feats_npy','feats_npy_sha256','feats_npy_size_bytes')):
            path=cache_root/info[name]
            if path.parent!=cache_root or info[name]!=path.name or str(path) in data_files:raise RuntimeError(prefix+'_CACHE_PATH')
            data_files[str(path)]={'sha256':info[sha],'bytes':info[bytekey],'kind':kind,'timeframe':tf}
    if len(data_files)!=28:raise RuntimeError(prefix+'_DATA_INVENTORY')
    for name,ref in data_files.items():
        path=verified({'path':name,'sha256':ref['sha256']},prefix+'_DATA')
        if 'bytes' in ref and path.stat().st_size!=ref['bytes']:raise RuntimeError(prefix+'_DATA_SIZE')
        ref['bytes']=path.stat().st_size;ref['whole_file_hash_passes']=1
    if sum(v['bytes'] for v in data_files.values())!=design['known_payload_bytes']['data_total']:
        raise RuntimeError(prefix+'_DATA_TOTAL_BYTES')
    # All18 source footers and10 NPY headers precede any numeric payload.
    rows={path:(_core_native_quote_footer(path,seal) if remaining_core else _technical_native_footer(path,seal)) for path in parts}
    for name,ref in data_files.items():
        if ref['kind'] in ('features','timestamps'):
            count=cache['tfs'][ref['timeframe']]['n_bars']
            _technical_npy_header(name,(count,253) if ref['kind']=='features' else (count,),
                'float32' if ref['kind']=='features' else 'int64')
    timestamps={}
    durations={'M5':5*MINUTE,'M15':15*MINUTE,'H1':60*MINUTE,'H4':240*MINUTE,'D1':DAY}
    for tf,info in cache['tfs'].items():
        clock=np.load(cache_root/info['ts_npy'],mmap_mode='r',allow_pickle=False)
        _projection_integer_clock(clock,prefix+'_CACHE_CLOCK')
        phase=(22*60*MINUTE)%durations[tf]
        if (clock[0]!=info['first_ts_ns'] or clock[-1]!=info['last_ts_ns'] or np.any((clock-phase)%durations[tf])
                or np.any(clock+durations[tf]>seal) or np.any(clock>=seal)):
            raise RuntimeError(prefix+'_CACHE_CLOSED_CLOCK')
        timestamps[tf]=clock
    required=('time','open','high','low','close','volume')+(NATIVE_CORE_QUOTE_FIELDS if remaining_core else ());chunks={key:[] for key in required};last=None;decoded=0
    for path in parts:
        table=pq.ParquetFile(path).read(columns=list(required),use_threads=False)
        if any(table[key].null_count for key in required):raise RuntimeError(prefix+'_SOURCE_NULL')
        clock=table['time'].cast(pa.int64()).to_numpy();year=Path(path).parent.name;bounds=native['year_time_bounds'][year]
        if (len(clock)!=rows[path] or len(clock)!=native['year_rows'][year] or not len(clock)
                or int(clock[0])!=pd.Timestamp(bounds['time_min_utc']).value or int(clock[-1])!=pd.Timestamp(bounds['time_max_utc']).value
                or (last is not None and clock[0]<=last)):raise RuntimeError(prefix+'_YEAR_CLOCK')
        last=int(clock[-1]);chunks['time'].append(clock);decoded+=len(clock)
        for key in required[1:]:chunks[key].append(table[key].to_numpy())
    m5={('bar_start_ns' if name=='time' else name):np.concatenate(values) for name,values in chunks.items()};del chunks,table
    if (decoded!=native['row_count'] or decoded!=design['source_contract']['native_rows']
            or int(m5['bar_start_ns'][0])!=pd.Timestamp(design['source_contract']['native_first_utc']).value):
        raise RuntimeError(prefix+'_FULL_HISTORY_ORIGIN')
    active={tf:set(profile['active_execution']['per_timeframe_fields'][tf])&set(fields) for tf in design['timeframes']}
    if any(len(names)!=active_count for names in active.values()):
        raise RuntimeError(prefix+('_COMPACT_470_SCOPE' if remaining_core else '_COMPACT_85_SCOPE' if trend_events or native_smc else '_COMPACT_55_SCOPE'))
    reports={};helper_reports={};selected_rows={}
    for tf in design['timeframes']:
        if remaining_core:
            info=cache['tfs'][tf]
            report,helper,count=_core_compare_owner_blocks(m5,tf,timestamps[tf],cache_root/info['feats_npy'],constants['level_expiry_bars'][tf])
            reports[tf]=report;helper_reports[tf]=helper;selected_rows[tf]=count
            continue
        raw=native_resample_calendar_frame(m5,tf)
        reference,bounds,helper=(reference_native_smc_block(raw) if native_smc else
            reference_trend_event_block(raw) if trend_events else reference_technical_block(raw))
        clock=timestamps[tf];positions=np.searchsorted(raw['bar_start_ns'],clock)
        if (np.any(positions>=len(raw['bar_start_ns'])) or not np.array_equal(raw['bar_start_ns'][positions],clock)
                or not np.array_equal(raw['bar_start_ns'][(raw['bar_start_ns']>=clock[0])&(raw['bar_start_ns']<=clock[-1])],clock)):
            raise RuntimeError(prefix+'_EXACT_CACHE_TIMESTAMP_MAPPING:'+tf)
        info=cache['tfs'][tf];matrix=np.load(cache_root/info['feats_npy'],mmap_mode='r',allow_pickle=False)
        target={name:np.asarray(matrix[:,j]) for name,j in zip(fields,columns)}
        ref={name:value[positions] for name,value in reference.items()};radii={name:value[positions] for name,value in bounds.items()}
        report=(compare_technical_reference(ref,target,radii,discrete_fields=NATIVE_SMC_DISCRETE_FIELDS) if native_smc else
            compare_technical_reference(ref,target,radii,discrete_fields=fields) if trend_events else compare_technical_reference(ref,target,radii))
        for name,stat in report.items():
            if not stat['numeric_reference_qualified']:raise RuntimeError(prefix+'_ACTIVE_LIBRARY_FIELD_UNOBSERVED:'+tf+':'+name)
            stat['compact_profile_active']=name in active[tf]
            if trend_events or native_smc:
                stat['positive_reference_values']=int(np.sum(ref[name]>0.))
                stat['zero_reference_values']=int(np.sum(ref[name]==0.))
        reports[tf]=report;helper_reports[tf]=helper;selected_rows[tf]=len(clock)
        del raw,reference,bounds,ref,radii,matrix,target
    for path,before in identities.items():
        if identity(Path(path).stat())!=before:raise RuntimeError(prefix+'_POST_INPUT_CHANGED')
    for path,digest in metadata_sha.items():io.verify_digest(Path(path),digest,prefix+'_POST_METADATA')
    for path,ref in runtime_files.items():io.verify_digest(Path(path),ref['sha256'],prefix+'_POST_RUNTIME')
    for name,digest in binding['code_sha256'].items():io.verify_digest(REPO/name,digest,prefix+'_POST_CODE')
    io.verify_digest(plan_path,binding['plan_sha256'],prefix+'_POST_PLAN');io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],prefix+'_POST_POLICY')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):
        raise RuntimeError(prefix+'_POST_SOURCE_CHANGED')
    result={'schema_version':'gx1_native_core_numeric_result_v1' if remaining_core else 'gx1_native_smc_numeric_result_v1' if native_smc else 'gx1_trend_event_numeric_result_v1' if trend_events else 'gx1_technical_numeric_result_v1',
        'status':'COMPLETE_SCOPED_94_REMAINING_ACTIVE_CORE_FORMULAS_ALL5TF' if remaining_core else 'COMPLETE_SCOPED_24_NATIVE_SMC_GEOMETRY_FORMULAS_ALL5TF' if native_smc else 'COMPLETE_SCOPED_17_TREND_EVENT_FORMULAS_ALL5TF' if trend_events else 'COMPLETE_SCOPED_23_TECHNICAL_FORMULAS_ALL5TF',
        'source_head':binding['source_head'],'source_unchanged':True,'plan_sha256':binding['plan_sha256'],'limits':proof,
        'design':plan['design'],'error_spec':plan['error_spec'],'references':refs,'runtime_files':runtime_files,
        'audited_formulas':len(fields),'timeframes':design['timeframes'],'qualified_output_fields':len(fields)*len(design['timeframes']),
        'compact_active_outputs':sum(map(len,active.values())),'compact_parked_outputs':sum(len(fields)-len(names) for names in active.values()),
        'fields':reports,'full_history_reference_reports':helper_reports,'native_rows_decoded':decoded,'native_projected_columns':list(required),
        'cache_rows_compared':selected_rows,'cache_feature_columns':list(columns),'cache_width':253,
        'data_files':data_files,'unique_data_files_hashed_once':28,'bound_payload_bytes':sum(x['bytes'] for x in data_files.values()),
        'physical_disk_io_bytes_measured':False,'prebuilt_payload_read_or_hashed':False,'full_history_origin':design['source_contract']['native_first_utc'],
        'native_sequence_or_PT_or_MBP_payload_read':False,'production_indicators_called_by_oracle':False,'normalization_fit_performed':False,
        'test_data_used':False,'model_forwards':0,'optimizer_updates':0,'learning_qualified':False,'economic_edge_qualified':False,
        'other_technical_families_qualified':False,'individual_learned_use_proven':False}
    if trend_events:
        result.update({'inherited_127_arithmetic_spec':spec['inherited_127_arithmetic_spec'],
            'design_cache_column_clarification':gate['design_cache_column_clarification'],
            'logical_selected_feature_bytes':clarification['logical_17selected_feature_bytes'],
            'previous_technical_numeric_review':refs['previous_technical_numeric_review'],
            'all_outputs_discrete_exact':True,'branch_ambiguity_policy':'fail_before_cache_comparison',
            'BB_or_ADX_or_other_technical_formulas_recomputed':False})
    if native_smc:
        result.update({'inherited_127_arithmetic_spec':spec['inherited_127_arithmetic_spec'],
            'qualified_mtf_numeric_index':refs['qualified_mtf_numeric_index'],
            'previous_technical_numeric_review':refs['previous_technical_numeric_review'],
            'previous_trend_numeric_review':refs['previous_trend_numeric_review'],
            'logical_selected_feature_bytes':design['known_payload_bytes']['logical_24selected_feature_bytes'],
            'discrete_fields':list(NATIVE_SMC_DISCRETE_FIELDS),'continuous_fields_per_TF':10,
            'pivot_comparison_tolerance':NATIVE_SMC_PIVOT_TOLERANCE,'BB_or_ADX_or_other_technical_formulas_recomputed':False})
    if remaining_core:
        result.update({'inherited_127_arithmetic_spec':spec['inherited_127_arithmetic_spec'],
            'qualified_mtf_numeric_index':refs['qualified_mtf_numeric_index'],
            'previous_native_smc_numeric_review':refs['previous_native_smc_numeric_review'],
            'logical_selected_feature_bytes':design['known_payload_bytes']['logical_94selected_feature_bytes'],
            'discrete_fields':list(design['discrete_fields']),'continuous_fields_per_TF':57,
            'level_expiry_bars':constants['level_expiry_bars'],'owner_block_order':['swing20','local27','level33','cross14'],
            'internal_float32_wires_propagated_without_target_cache_input':True,
            'owner_blocks_compared_and_released_sequentially':True,
            'prior_320_cached_output_surfaces_reaudited':False,'BB_or_ADX_output_surfaces_recomputed':False,
            'qualified_primitives_reused_for_new_formulas':['WilderATR14','EMA','raw_native_SMC_n3_replay','native_calendar_axis']})
    output=staging/(prefix+'_RESULT.json');io.write_json(output,result)
    io.write_json(staging/(prefix+'_TERMINAL.json'),{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(output)})
    publish_bundle_directory_noreplace(staging,root/(prefix+'_RESULTS'));print(prefix+'_REFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--binding',type=Path);group.add_argument('--evaluation-binding',type=Path)
    group.add_argument('--gc-event-numeric-binding',type=Path)
    group.add_argument('--gc-event-book-numeric-binding',type=Path)
    group.add_argument('--gc-trade-numeric-binding',type=Path)
    group.add_argument('--gc-smc-numeric-binding',type=Path)
    group.add_argument('--gc-profile-numeric-binding',type=Path)
    group.add_argument('--gc-basis-numeric-binding',type=Path)
    group.add_argument('--gc-native-projection-binding',type=Path)
    group.add_argument('--technical-numeric-binding',type=Path)
    group.add_argument('--trend-events-numeric-binding',type=Path)
    group.add_argument('--native-smc-numeric-binding',type=Path)
    group.add_argument('--native-core-numeric-binding',type=Path)
    group.add_argument('--p0-diagnosis-binding',type=Path);group.add_argument('--p0-review-binding',type=Path);group.add_argument('--l1-binding',type=Path);group.add_argument('--cash-binding',type=Path);group.add_argument('--p0-preparation-binding',type=Path);args=parser.parse_args()
    if args.native_core_numeric_binding is not None:run_technical_numeric(args.native_core_numeric_binding,remaining_core=True)
    elif args.native_smc_numeric_binding is not None:run_technical_numeric(args.native_smc_numeric_binding,native_smc=True)
    elif args.trend_events_numeric_binding is not None:run_technical_numeric(args.trend_events_numeric_binding,trend_events=True)
    elif args.technical_numeric_binding is not None:run_technical_numeric(args.technical_numeric_binding)
    elif args.gc_native_projection_binding is not None:run_gc_native_projection_numeric(args.gc_native_projection_binding)
    elif args.gc_basis_numeric_binding is not None:run_gc_event_numeric(args.gc_basis_numeric_binding,basis=True)
    elif args.gc_profile_numeric_binding is not None:run_gc_event_numeric(args.gc_profile_numeric_binding,profile=True)
    elif args.gc_smc_numeric_binding is not None:run_gc_event_numeric(args.gc_smc_numeric_binding,smc=True)
    elif args.gc_trade_numeric_binding is not None:run_gc_event_numeric(args.gc_trade_numeric_binding,trade=True)
    elif args.gc_event_book_numeric_binding is not None:run_gc_event_numeric(args.gc_event_book_numeric_binding,event_book=True)
    elif args.gc_event_numeric_binding is not None:run_gc_event_numeric(args.gc_event_numeric_binding)
    elif args.binding is not None:run(args.binding)
    elif args.evaluation_binding is not None:run_evaluation(args.evaluation_binding)
    elif args.l1_binding is not None:run_l1_quality(args.l1_binding)
    elif args.cash_binding is not None:run_cash_review(args.cash_binding)
    elif args.p0_diagnosis_binding is not None:run_p0_diagnosis(args.p0_diagnosis_binding)
    elif args.p0_review_binding is not None:run_p0_review(args.p0_review_binding)
    else:run_p0_preparation(args.p0_preparation_binding)
