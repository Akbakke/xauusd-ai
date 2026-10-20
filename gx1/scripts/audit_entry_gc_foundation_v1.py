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


def read_event_numeric_columns(path, columns, start, end, seal):
    """Inspect all selected footer boundaries before decoding any payload."""
    parquet = pq.ParquetFile(path); clock = parquet.schema_arrow.names.index('bar_start_ns')
    groups = []
    for row in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row).column(clock).statistics
        if stats is None or not stats.has_min_max:
            raise RuntimeError('GC_EVENT_NUMERIC_ROW_GROUP_CLOCK_UNPROVEN')
        low, high = int(stats.min), int(stats.max)
        if high < start or low >= end:
            continue
        if high >= seal:
            raise RuntimeError('GC_EVENT_NUMERIC_ROW_GROUP_INTERSECTS_TEST')
        groups.append(row)
    if not groups:
        raise RuntimeError('GC_EVENT_NUMERIC_NO_SAFE_ROW_GROUPS')
    table = parquet.read_row_groups(groups, columns=columns, use_threads=False)
    times = table['bar_start_ns'].to_numpy()
    if np.any(times >= seal):
        raise RuntimeError('GC_EVENT_NUMERIC_TEST_CLOCK')
    take = (times >= start) & (times < end)
    values = {name: table[name].to_numpy()[take] for name in columns}
    return values, groups


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


def run_gc_event_numeric(binding_path, *, event_book=False, trade=False):
    plan_id = GC_TRADE_PLAN_ID if trade else GC_EVENT_BOOK_PLAN_ID if event_book else GC_EVENT_NUMERIC_PLAN_ID
    policy_key = GC_TRADE_POLICY_KEY if trade else GC_EVENT_BOOK_POLICY_KEY if event_book else GC_EVENT_NUMERIC_POLICY_KEY
    root = Path('/home/andre2/GX1_RUNS')/plan_id
    prefix = 'GC_TRADE_NUMERIC' if trade else 'GC_EVENT_BOOK_NUMERIC' if event_book else 'GC_EVENT_NUMERIC'
    schema = 'gx1_gc_trade_numeric_audit_plan_v1' if trade else 'gx1_gc_event_book_numeric_audit_plan_v1' if event_book else 'gx1_gc_event_numeric_audit_plan_v1'
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
            or (trade and start != STUDY_START_NS)):
        raise RuntimeError('GC_EVENT_NUMERIC_SCAN_BOUNDARY')
    publication = binding_read(plan['publication_terminal'])
    source_root = Path(plan['publication_terminal']['path']).parent
    if (source_root != Path('/home/andre2/GX1_RUNS/NATIVE_GC_TRUSTED_RECORD_RESTORATION_20261018_024/RESULTS')
            or Path(plan['publication_terminal']['path']).name != 'TERMINAL.json'
            or publication['status'] != 'COMPLETE' or publication['exit_code'] != 0
            or publication['source_unchanged'] is not True or publication['test_data_used'] is not False):
        raise RuntimeError('GC_EVENT_NUMERIC_PUBLICATION')
    inputs = plan['inputs']; tfs = ('M5','M15','H1','H4','D1')
    keys = {'m1_trade','m1_trade_sizes','timeframes'} if trade else {'m1_events','m1_book','strict_m5_book','timeframes'} if event_book else {'m1_events','timeframes'}
    if set(inputs) != keys or set(inputs['timeframes']) != set(tfs):
        raise RuntimeError('GC_EVENT_NUMERIC_INPUT_INVENTORY')
    files = {'M1_GC_INPUTS.parquet':inputs['m1_trade'],
        'M1_TRADE_SIZE_STATISTICS_UNMASKED.parquet':inputs['m1_trade_sizes']} if trade else {'M1_EVENT_DIAGNOSTICS.parquet': inputs['m1_events']}
    files.update({'TRADE_FEATURES_'+tf+'.parquet': inputs['timeframes'][tf] for tf in tfs})
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
    common = ['bar_start_ns','available_ns','contract_index']
    source_columns = common + ['qualified_final_event_count'] + [d+s for d in GC_EVENT_CLOCK_DOMAINS for s in ('_count','_sum_ns')]
    if event_book:source_columns=common+list(GC_EVENT_REFERENCE_REDUCERS)+['qualified_covered_ns']
    if trade:
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
        if trade:
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
        if event_book or trade:columns=list(reference)
        actual, groups = read_event_numeric_columns(Path(inputs['timeframes'][tf]['path']),columns,start,end,seal)
        keep = actual['available_ns'] <= end; actual = {name:value[keep] for name,value in actual.items()}
        if trade:
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
    output = staging/(prefix+'_RESULT.json'); io.write_json(output,result)
    io.write_json(staging/(prefix+'_TERMINAL.json'), {'status':'COMPLETE','exit_code':0,
        'source_unchanged':True,'test_data_used':False,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'result_sha256':io.sha256_file(output)})
    publish_bundle_directory_noreplace(staging,root/(prefix+'_RESULTS'))
    print(prefix+'_REFERENCE_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--binding',type=Path);group.add_argument('--evaluation-binding',type=Path)
    group.add_argument('--gc-event-numeric-binding',type=Path)
    group.add_argument('--gc-event-book-numeric-binding',type=Path)
    group.add_argument('--gc-trade-numeric-binding',type=Path)
    group.add_argument('--p0-diagnosis-binding',type=Path);group.add_argument('--p0-review-binding',type=Path);group.add_argument('--l1-binding',type=Path);group.add_argument('--cash-binding',type=Path);group.add_argument('--p0-preparation-binding',type=Path);args=parser.parse_args()
    if args.gc_trade_numeric_binding is not None:run_gc_event_numeric(args.gc_trade_numeric_binding,trade=True)
    elif args.gc_event_book_numeric_binding is not None:run_gc_event_numeric(args.gc_event_book_numeric_binding,event_book=True)
    elif args.gc_event_numeric_binding is not None:run_gc_event_numeric(args.gc_event_numeric_binding)
    elif args.binding is not None:run(args.binding)
    elif args.evaluation_binding is not None:run_evaluation(args.evaluation_binding)
    elif args.l1_binding is not None:run_l1_quality(args.l1_binding)
    elif args.cash_binding is not None:run_cash_review(args.cash_binding)
    elif args.p0_diagnosis_binding is not None:run_p0_diagnosis(args.p0_diagnosis_binding)
    elif args.p0_review_binding is not None:run_p0_review(args.p0_review_binding)
    else:run_p0_preparation(args.p0_preparation_binding)
