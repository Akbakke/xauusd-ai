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

def qualify_new_fl_p0(plan,stage):
    from gx1.scripts import research_m5_intraday_recheck_v1 as m5
    from gx1.scripts.research_openbb_source_review_v1 import known_cot_delay_diagnostic_v1
    from gx1.contracts import entry_research_evaluation_v1 as ev
    from gx1.contracts.xau_tape_provenance_v1 import scheduled_weekly_closure_mask
    from gx1.scripts import research_ta_campaign_v1 as macro
    features=pq.read_table(binding_read(plan['features'])).to_pandas();starts=features.m5_bar_start_ns.to_numpy();decisions=features.decision_ns.to_numpy()
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
    out={'new_F_L_joint_input_rows':int(fl.sum()),'known_old_COT_availability_unsafe_rows':int(unsafe.sum()),
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
        out['horizons'][str(h)]={'v2_execution_valid':int(ex.valid.sum()),'exact_M1_path_valid':int(strict['valid'].sum()),'old_valid_with_delayed_terminal':int(np.count_nonzero(ex.valid&(strict['terminal_delay_ns']>0))),'old_valid_with_interior_gap':int(np.count_nonzero(ex.valid&(strict['interior_gap_intervals']>0))),'new_FL_and_exact_path':int(candidate.sum()),'after_known_COT_leak_exclusion':int(without_known_leak.sum()),'same_cost_model_all_eight_arms':True,'cost_sensitivity':sensitivity}
        del central
    masks.to_parquet(stage/'P0_PREPARATION_MASKS.parquet',index=False)
    out.update(models_fitted=False,TEST_accessed=False,P0_training_admitted=False,baseline_Exit_or_native_targets_changed=False,
       blockers=prep['remaining_admission_gates'],native_D1_h20_qualification_separate=True)
    return out


def run_p0_preparation(binding_path):
    identity='GC_P0_COHORT_PREPARATION_20261017_018';root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/identity
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('P0_PREP_BINDING')
    binding=io.read_json(binding_path);rel='configs/research/'+identity+'.json';io.verify_digest(REPO/rel,binding['plan_sha256'],'P0_PREP_PLAN');plan=io.read_json(REPO/rel)
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_PREP_POLICY');io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_PREP_APPROVAL')
    policy=io.read_json(REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_P0_cohort_preparation_018'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or policy['training_enabled'] is not False or approval['max_invocations']!=1 or approval['approved_plan_sha256']!=binding['plan_sha256'] or io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)):raise RuntimeError('P0_PREP_AUTHORITY')
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_PREP_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry();stage=root/'.RESULTS.staging';stage.mkdir(exist_ok=False)
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'source_head':binding['source_head']})
    result=qualify_new_fl_p0(plan,stage);result['source_head']=binding['source_head'];result['limits']=proof
    for p,h in binding['code_sha256'].items():io.verify_digest(REPO/p,h,'P0_PREP_CODE_POST')
    io.verify_digest(REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_PREP_POLICY_POST')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True):raise RuntimeError('P0_PREP_SOURCE_CHANGED')
    io.write_json(stage/'RESULT.json',result);io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':io.sha256_file(stage/'RESULT.json')})
    publish_bundle_directory_noreplace(stage,root/'RESULTS');print('GC_P0_COHORT_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--binding',type=Path);group.add_argument('--evaluation-binding',type=Path)
    group.add_argument('--l1-binding',type=Path);group.add_argument('--cash-binding',type=Path);group.add_argument('--p0-preparation-binding',type=Path);args=parser.parse_args()
    if args.binding is not None:run(args.binding)
    elif args.evaluation_binding is not None:run_evaluation(args.evaluation_binding)
    elif args.l1_binding is not None:run_l1_quality(args.l1_binding)
    elif args.cash_binding is not None:run_cash_review(args.cash_binding)
    else:run_p0_preparation(args.p0_preparation_binding)
