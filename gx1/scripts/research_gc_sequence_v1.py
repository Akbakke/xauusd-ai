"""Capped qualification of GC sequences; separate single-use CPU pilot gate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gx1.scripts import gc_source_disposition_v1 as io
from gx1.scripts import research_m5_intraday_recheck_v1 as old
from gx1.contracts import gc_sequence_research_v1 as contract
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace

PLAN=io.REPO/'configs/research/GC_SEQUENCE_ALL_FEATURES_20261017_027.json'
GATE='gc_sequence_research_20261017'


def load_inputs(spec):
    previous=io.read_json(old.require_binding(spec['previous_recipe']))
    for value in previous['inputs'].values():old.require_binding(value)
    source=io.read_json(Path(previous['inputs']['native_source_plan']['path']))
    if set(source['native_datasets'])!={'train','val'}:raise RuntimeError('GC_SEQUENCE_SPLIT_SEAL')
    old.require_binding(source['feature_inputs']['parquet'])
    base=pq.read_table(source['feature_inputs']['parquet']['path'])
    times,snap,ctx,session,fields,ctx_names=old.load_gc_native_rows(source,base)
    d=times+old.DECISION_OFFSET_NS
    if d.max()>=pd.Timestamp(spec['TEST_seal_utc']).value:raise RuntimeError('GC_SEQUENCE_TEST_SEAL')
    f=pq.read_table(previous['inputs']['F']['path']).to_pandas()
    l=pq.read_table(previous['inputs']['L']['path']).to_pandas()
    for table in (f,l):
        if not np.array_equal(d,table.decision_ns.to_numpy()):raise RuntimeError('GC_SEQUENCE_FEATURE_CLOCK')
    if not np.array_equal(d,base['decision_ns'].to_numpy()):raise RuntimeError('GC_SEQUENCE_BASE_CLOCK')
    c=f.contract_index.to_numpy()
    if not np.array_equal(c,l.contract_index.to_numpy()):
        raise RuntimeError('GC_SEQUENCE_CONTRACT_ALIGNMENT')
    for j,n in enumerate(ctx_names):
        if not np.array_equal(snap[:,fields.index('ctx_cont.'+n)],ctx[:,j]):raise RuntimeError('GC_SEQUENCE_CONTEXT_ALIAS:'+n)
    if [n for n in fields if n not in spec['excluded_macro_fields']]!=spec['fields']['native']:raise RuntimeError('GC_SEQUENCE_NATIVE_COVERAGE')
    raw_names=spec['fields']['F']+spec['fields']['L']+spec['fields']['native']+spec['fields']['basis']+spec['fields']['trade_activity']+['ctx_cat.session_id']
    if raw_names!=spec['raw_field_order']:raise RuntimeError('GC_SEQUENCE_RAW_FIELD_ORDER')
    raw=np.column_stack([f[n].to_numpy() for n in spec['fields']['F']]+[l[n].to_numpy() for n in spec['fields']['L']]+
        [snap[:,fields.index(n)] for n in spec['fields']['native']]+[base[n].to_numpy() for n in spec['fields']['basis']+spec['fields']['trade_activity']]+[session])
    x,encoded_names=contract.encode_inputs(raw,raw_names,spec['categorical_domains'])
    if encoded_names!=spec['encoded_field_order']:raise RuntimeError('GC_SEQUENCE_ENCODED_FIELD_ORDER')
    observed=f.ofi_valid.to_numpy()&f.bbo_valid.to_numpy()&l.rolling15_valid.to_numpy()
    observed &= base['m5_volume_valid'].to_numpy()&np.isfinite(x).all(axis=1)
    contract.require_contract_alignment(c,base['m5_contract_index'].to_numpy(),observed)
    windows=contract.sequence_rows(d,c,observed,length=spec['sequence_length'],step_ns=spec['step_seconds']*old.NS)
    for b in source['execution_tape_partitions']:old.require_binding(b)
    actual={str(p) for p in Path(source['reference']['data']['execution_tape']['root']).glob('year=*/*.parquet') if int(p.parent.name.split('=')[1])>=2024}
    if actual!={b['path'] for b in source['execution_tape_partitions']}:raise RuntimeError('GC_SEQUENCE_UNBOUND_TAPE')
    tape=old.load_tape(source['reference'],2024)
    ex=old.execution_for_horizon_v2(d,tape,horizon_minutes=spec['horizon_minutes'],latency_seconds=spec['latency_seconds'])
    valid=old.exact_intraday_mask_v3(d,tape,ex,horizon_minutes=spec['horizon_minutes'])['valid']
    signed=base['m5_signed_valid'].to_numpy()&np.isfinite(base['m5_delta_contracts'].to_numpy())
    audit=audit_features(spec,d,x,windows,f,l,base,tape)
    return previous,d,x,c,observed,windows,tape,ex,valid,signed,audit


def audit_features(spec,d,x,windows,f,l,base,tape):
    rows=np.unique(windows);minute=pq.read_table(old.require_binding(spec['F_minute'])).to_pandas()
    wanted=d[rows,None]-5*old.MINUTE_NS+np.arange(5)*old.MINUTE_NS
    idx=np.searchsorted(minute.bar_start_ns.to_numpy(),wanted)
    if np.any(idx>=len(minute)) or not np.array_equal(minute.bar_start_ns.to_numpy()[idx],wanted):raise RuntimeError('GC_FEATURE_MINUTE_CLOCK')
    checks={}
    for j,name in enumerate(spec['fields']['F']):
        values=minute[name].to_numpy()[idx]
        expected=values.sum(axis=1) if j<6 else values.mean(axis=1)
        if name=='mid_response_ticks':expected=minute.closing_mid_tick.to_numpy()[idx[:,-1]]-minute.opening_mid_tick.to_numpy()[idx[:,0]]
        if not np.array_equal(expected,f[name].to_numpy()[rows]):raise RuntimeError('GC_FEATURE_F_PRODUCER:'+name)
        checks[name]={'exact_M1_aggregation_rows':len(rows)}
    cache=pq.read_table(old.require_binding(spec['full_history_M1']),columns=['bar_start_ns','close_tick','volume','trades']).to_pandas()
    last=np.searchsorted(cache.bar_start_ns.to_numpy(),d[rows]-old.MINUTE_NS)
    if np.any(last>=len(cache)) or not np.array_equal(cache.bar_start_ns.to_numpy()[last],d[rows]-old.MINUTE_NS):raise RuntimeError('GC_FEATURE_CLOSE_CLOCK')
    close=cache.close_tick.to_numpy()[last].astype(float)*spec['tick_usd']
    starts=np.searchsorted(cache.bar_start_ns.to_numpy(),wanted)
    if not np.array_equal(cache.bar_start_ns.to_numpy()[starts],wanted):raise RuntimeError('GC_FEATURE_ACTIVITY_CLOCK')
    for name,source_name in [('m5_volume','volume'),('m5_trades','trades')]:
        expected=cache[source_name].to_numpy()[starts].sum(axis=1)
        if not np.array_equal(expected,base[name].to_numpy()[rows]):raise RuntimeError('GC_FEATURE_ACTIVITY_VALUE:'+name)
        checks[name]={'exact_real_M1_sum_rows':len(rows)}
    if not np.array_equal(close,base['m5_close_usd'].to_numpy()[rows]):raise RuntimeError('GC_FEATURE_CLOSE_CONSERVATION')
    prior=rows-1;adjacent=(rows>0)&(d[rows]-d[np.maximum(prior,0)]==old.DECISION_OFFSET_NS)
    for level in ('vwap','poc','val','vah'):
        price=l['rolling15_'+level+'_usd'].to_numpy()
        distance=(close-price[rows])/spec['tick_usd'];name='rolling15_'+level+'_distance_ticks'
        if not np.array_equal(distance,l[name].to_numpy()[rows]):raise RuntimeError('GC_FEATURE_LEVEL_DISTANCE:'+name)
        checks[name]={'exact_rows':len(rows)}
        expected=(close[adjacent]-price[prior[adjacent]])/spec['tick_usd'];name='rolling15_'+level+'_prior_level_response_ticks'
        if not np.array_equal(expected,l[name].to_numpy()[rows[adjacent]]):raise RuntimeError('GC_FEATURE_PRIOR_LEVEL:'+name)
        checks[name]={'exact_adjacent_native_rows':int(adjacent.sum()),'nonadjacent_rows_inherit_frozen_producer_proof':int((~adjacent).sum())}
    ti=np.searchsorted(tape.time_ns,d[rows]-old.MINUTE_NS)
    if not np.array_equal(tape.time_ns[ti],d[rows]-old.MINUTE_NS):raise RuntimeError('GC_FEATURE_BASIS_CLOCK')
    mid=(tape.bid_close[ti]+tape.ask_close[ti])/2;basis=1e4*(close/mid-1)
    if not np.array_equal(basis,base['m5_spot_basis_bps'].to_numpy()[rows]):raise RuntimeError('GC_FEATURE_BASIS_VALUE')
    ledger=io.read_json(old.require_binding(spec['native_field_ledger']))
    legacy={v['name']:v for v in ledger['fields'] if v['surface']=='signal'}
    if set(legacy)!=set(spec['fields']['native']):raise RuntimeError('GC_FEATURE_NATIVE_LEDGER_COVERAGE')
    families={n:sum(v['specialist']==n for v in legacy.values()) for n in sorted({v['specialist'] for v in legacy.values()})}
    if len(families)!=8:raise RuntimeError('GC_FEATURE_NATIVE_FAMILY_COVERAGE')
    columns={}
    for j,name in enumerate(spec['encoded_field_order']):
        values=x[rows,j]
        if not np.isfinite(values).all():raise RuntimeError('GC_FEATURE_CONSUMED_NONFINITE:'+name)
        columns[name]={'rows':len(rows),'min':float(values.min()),'max':float(values.max()),'std':float(values.std()),'distinct':len(np.unique(values))}
    return {'status':'ALL_CONSUMED_FIELDS_CHECKED','physical_input_rows':len(rows),'native_fields':len(legacy),
        'native_families':families,'encoded_fields':len(columns),'columns':columns,'GC_formula_checks':checks,'basis_exact_rows':len(rows),
        'native_formula_proof':'reused prior field ledger and corrected-generation receipt; current row identity/finite/domain/alias checks, not a fresh raw reconstruction of every native formula',
        'constant_consumed_fields':[n for n,v in columns.items() if v['distinct']==1],
        'excluded_fields':{'macro':spec['excluded_macro_fields'],'signed_flow':'insufficient strict contiguous coverage; retained diagnostics'},'test_data_used':False}


def native_inventory(spec):
    review=io.read_json(old.require_binding(spec['native_review']))
    receipt=io.read_json(old.require_binding(spec['native_receipt']))
    for key in ('training_state','training_pointer','recipe'):old.require_binding(review[key])
    corrected=io.read_json(old.require_binding(spec['corrected_result']))
    candidate=pq.read_table(old.require_binding(spec['corrected_h20']),columns=['strict_terminal_valid','strict_long_net_bps','strict_short_net_bps','strict_flat_bps','terminal_delay_seconds'])
    ok=candidate['strict_terminal_valid'].to_numpy()
    for name in ('strict_long_net_bps','strict_short_net_bps','strict_flat_bps'):
        values=candidate[name].to_numpy()
        if not np.isfinite(values[ok]).all() or not np.isnan(values[~ok]).all():raise RuntimeError('GC_SEQUENCE_H20_CENSORING')
    if not np.all(candidate['terminal_delay_seconds'].to_numpy()[ok]==0):raise RuntimeError('GC_SEQUENCE_H20_DELAY')
    recipe=io.read_json(Path(review['recipe']['path']))
    metadata=io.read_json(old.require_binding(recipe['files']['source_bundle_metadata']))
    from gx1.contracts.entry_exit_production_architecture_v1 import current_entry_exit_architecture_observation
    architecture=current_entry_exit_architecture_observation()
    return {'last_training_started_utc':receipt['started_utc'],'last_training_finished_utc':receipt['finished_utc'],
        'optimizer_steps':review['optimizer_steps'],'unique_train_rows':review['unique_train_rows'],
        'loss_horizon_trading_days':review['loss_horizon_trading_days'],'learned_exit_teacher_used':False,
        'unchanged_exit_state_fields':review['unchanged_exit_state_fields'],'model_state_sha256':review['final_model_state_sha256'],
        'checkpoint':review['training_state'],'recipe':review['recipe'],'source_commit':review['source_commit'],
        'learning_admitted':review['learning_admitted'],'economic_rollout':review['economic_rollout'],
        'control_h20_before':review['observed_horizons']['control_0']['20'],
        'control_h20_after':review['observed_horizons']['control_4071']['20'],
        'architecture':architecture,'input_metadata':metadata,'corrected_h20_rows':len(ok),'corrected_h20_valid_rows':int(ok.sum()),'corrected_target_result_keys':list(corrected),'corrected_h20':spec['corrected_h20'],
        'native_new_fit_blockers':spec['native_blockers'],'baseline_modified':False}


def model_preflight(x,spec):
    import torch
    from gx1.models.gc_sequence_research_v1 import GCSequenceTransformer,GCSequenceTCN
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);output={}
    for name,cls in [('transformer',GCSequenceTransformer),('tcn',GCSequenceTCN)]:
        m=cls(x.shape[2],x.shape[1],spec[name]).eval();start=time.monotonic()
        with torch.no_grad():p=m(torch.from_numpy(x))
        forward_seconds=time.monotonic()-start
        if not torch.isfinite(p).all():raise RuntimeError('GC_SEQUENCE_FORWARD_NONFINITE')
        weights=[v.detach().clone() for v in m.parameters()]
        actual=torch.from_numpy(x.copy()).requires_grad_(True);m(actual).sum().backward()
        gradient=actual.grad.abs().sum(dim=(0,1))
        if not torch.isfinite(gradient).all() or torch.any(gradient==0):raise RuntimeError('GC_SEQUENCE_DISCONNECTED_INPUT')
        if not all(torch.equal(a,b) for a,b in zip(weights,m.parameters())):raise RuntimeError('GC_SEQUENCE_PREFLIGHT_WEIGHT_MUTATION')
        output[name]={'parameters':sum(p.numel() for p in m.parameters()),'real_rows':len(x),
            'forward_seconds_single_cold_batch':forward_seconds,'optimizer_steps':0,'trained':False,'input_channels_with_nonzero_derivative':int((gradient>0).sum()),'weights_unchanged':True}
    return output


def qualify(spec,stage):
    previous,d,x,c,observed,windows,tape,ex,valid,signed,audit=load_inputs(spec)
    rows=windows[:,-1];sd=d[rows];outcomes=ex.exit_ns[rows];target_ok=valid[rows]
    months=pd.DatetimeIndex(d,tz='UTC').strftime('%Y-%m').to_numpy()
    coverage={m:{'population':int((months==m).sum()),'single_window':int((observed&(months==m)).sum()),
        'sequences':int((months[rows]==m).sum())} for m in sorted(set(months))}
    folds={}
    for f in spec['folds']:
        tr,ev=contract.purged_fold(sd,outcomes,target_ok,f,embargo_seconds=spec['embargo_seconds'])
        folds[f['fold_id']]={'train_rows':len(tr),'evaluation_input_rows':len(ev),'evaluation_metric_rows':int(target_ok[ev].sum()),
            'train_rows_sha256':hashlib.sha256(rows[tr].tobytes()).hexdigest(),
            'evaluation_rows_sha256':hashlib.sha256(rows[ev].tobytes()).hexdigest()}
    first=spec['folds'][0];tr,ev=contract.purged_fold(sd,outcomes,target_ok,first,embargo_seconds=spec['embargo_seconds'])
    if not len(tr) or not len(ev):raise RuntimeError('GC_SEQUENCE_EMPTY_COHORT')
    mean,scale,unique=contract.train_normalization(x,windows,tr)
    probe=contract.standardized(x,windows[tr[:spec['batch_size']]],mean,scale)
    preflight=model_preflight(probe,spec)
    predictions=pd.read_parquet(old.require_binding(spec['previous_predictions']))
    if predictions.decision_ns.duplicated().any():raise RuntimeError('GC_SEQUENCE_DUPLICATE_FROZEN_REFERENCE')
    eval_all=np.flatnonzero((sd>=pd.Timestamp(spec['folds'][0]['validation_start_utc_inclusive']).value)&(sd<pd.Timestamp(spec['folds'][-1]['validation_end_utc_exclusive']).value))
    matched=predictions.set_index('decision_ns').reindex(sd[eval_all])
    if matched[['tree_CORE','tree_FLV','y30_mid_bps']].isna().any().any():raise RuntimeError('GC_SEQUENCE_REFERENCE_ROWS')
    if not np.array_equal(matched.y30_mid_bps.to_numpy(),ex.y_mid_bps[rows[eval_all]]):raise RuntimeError('GC_SEQUENCE_REFERENCE_TARGET')
    np.savez(stage/'INPUTS.npz',features=x,windows=windows,decisions=sd,outcomes=outcomes,valid=target_ok,
        y=ex.y_mid_bps[rows],train=tr,evaluation=ev,mean=mean,scale=scale,normalization_rows=unique)
    pd.DataFrame({'decision_ns':sd,'physical_row':rows,'target_valid':target_ok,'contract_index':c[rows]}).to_parquet(stage/'COHORT.parquet',index=False)
    matched.reset_index().to_parquet(stage/'FROZEN_REFERENCES.parquet',index=False)
    io.write_json(stage/'FEATURE_AUDIT.json',audit)
    basis=x[rows,spec['encoded_field_order'].index('m5_spot_basis_bps')]
    return {'status':'QUALIFIED_RESEARCH_INPUTS_NOT_TRAINING_ADMISSION' if len(tr)>=spec['minimum_train_rows'] else 'QUALIFICATION_COMPLETE_INSUFFICIENT_PILOT_SUPPORT','pilot_sample_floor_met':len(tr)>=spec['minimum_train_rows'],'observed_real_data':True,
        'feature_audit':{'native_fields':audit['native_fields'],'native_families':audit['native_families'],'constant_consumed_fields':audit['constant_consumed_fields']},'native':native_inventory(spec),'population_rows':len(d),'qualified_single_windows':int(observed.sum()),
        'complete_sequences':len(windows),'sequence_shape':[len(windows),spec['sequence_length'],x.shape[1]],
        'all_tokens_observed':True,'numeric_imputation_used':False,'one_contract_per_sequence':True,
        'sequence_gap_or_roll_crossings':0,'max_input_time_equals_decision':bool(np.all(d[windows[:,-1]]==sd)),
        'source_clock_scope':'existing receive-clock/reset/gap/seam-qualified F and rolling15 L masks reused; no new raw audit',
        'strict_signed_sequences':int(signed[windows].all(axis=1).sum()),
        'signed_delta_used_in_primary_model':False,'signed_delta_reason':'separate coverage diagnostic; no invented missing aggressor values',
        'basis_bps':{'min':float(basis.min()),'median':float(np.median(basis)),'max':float(basis.max())},
        'basis_scope':'observed GC/XAU close-basis covariate; executable economics always XAU bid/ask',
        'monthly_coverage':coverage,'folds':folds,'model_preflight':preflight,
        'normalization_fit_rows':len(unique),'normalization_fit_role':'unique physical rows in first purged TRAIN input windows',
        'pilot_training_authorized':False,'test_data_used':False,'models_fitted':0,'independent_OOS':False,
        'limitations':spec['limitations']}


def fit_network(name,x,y,train,evaluation,spec,stage):
    import torch
    from gx1.models.gc_sequence_research_v1 import GCSequenceTransformer,GCSequenceTCN
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);torch.use_deterministic_algorithms(True)
    cls={'transformer':GCSequenceTransformer,'tcn':GCSequenceTCN}[name]
    model=cls(x.shape[2],x.shape[1],spec[name]);opt=torch.optim.AdamW(model.parameters(),
        lr=spec['optimizer']['learning_rate'],weight_decay=spec['optimizer']['weight_decay'])
    center=float(y[train].mean());scale=float(y[train].std())
    if scale<=0:raise RuntimeError('GC_SEQUENCE_CONSTANT_TARGET')
    tx=torch.from_numpy(x);ty=torch.from_numpy(((y-center)/scale).astype(np.float32))
    model.eval()
    with torch.no_grad():
        initial_train_mse=float(((model(tx[train]).numpy()*scale+center-y[train])**2).mean())
    start=time.monotonic();losses=[]
    for step in range(spec['optimizer_steps']):
        if time.monotonic()-start>spec['network_max_wall_seconds']:raise RuntimeError('GC_SEQUENCE_NETWORK_BUDGET')
        b=train[(np.arange(spec['batch_size'])+step*spec['batch_size'])%len(train)]
        model.train();opt.zero_grad(set_to_none=True);loss=((model(tx[b])-ty[b])**2).mean()
        if not torch.isfinite(loss):raise RuntimeError('GC_SEQUENCE_NONFINITE_LOSS')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec['optimizer']['grad_clip'],error_if_nonfinite=True)
        opt.step();losses.append(float(loss.detach()))
    fit_seconds=time.monotonic()-start;model.eval();start=time.monotonic();pred=[]
    with torch.no_grad():
        for i in range(0,len(evaluation),spec['batch_size']):pred.append(model(tx[evaluation[i:i+spec['batch_size']]]).numpy()*scale+center)
    pred=np.concatenate(pred);inference_seconds=time.monotonic()-start
    with torch.no_grad():final_train_mse=float(((model(tx[train]).numpy()*scale+center-y[train])**2).mean())
    torch.save({'model':model.state_dict(),'target_mean':center,'target_scale':scale},stage/(spec['artifact_prefix']+'_'+name+'.pt'))
    restored=cls(x.shape[2],x.shape[1],spec[name]).eval()
    saved=torch.load(stage/(spec['artifact_prefix']+'_'+name+'.pt'),map_location='cpu',weights_only=True);restored.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():
        batch=tx[evaluation[:spec['batch_size']]]
        if not torch.equal(model(batch),restored(batch)):raise RuntimeError('GC_SEQUENCE_MODEL_READBACK')
    return pred,{'optimizer_steps':len(losses),'first_batch_loss':losses[0],'last_batch_loss':losses[-1],
        'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,'inference_rows':len(evaluation),'initial_train_mse_bps2':initial_train_mse,'final_train_mse_bps2':final_train_mse,'model_readback_bit_exact':True,'unique_train_rows_seen':min(len(train),spec['optimizer_steps']*spec['batch_size'])}


def fit_fold(spec,stage,data,refs,fold):
    tr,ev=contract.purged_fold(data['decisions'],data['outcomes'],data['valid'],fold,embargo_seconds=spec['embargo_seconds'])
    if len(tr)<spec['minimum_train_rows']:raise RuntimeError('GC_SEQUENCE_PILOT_COHORT_TOO_SMALL')
    mean,scale,physical=contract.train_normalization(data['features'],data['windows'],tr)
    x=contract.standardized(data['features'],data['windows'],mean,scale);y=data['y'];prefix=fold['fold_id']
    np.savez(stage/(prefix+'_NORMALIZATION.npz'),mean=mean,scale=scale,physical_rows=physical,train=tr,evaluation=ev)
    locked=refs.set_index('decision_ns').reindex(data['decisions'][ev])
    if locked[['tree_CORE','tree_FLV']].isna().any().any():raise RuntimeError('GC_SEQUENCE_FOLD_REFERENCE')
    frame=pd.DataFrame({'decision_ns':data['decisions'][ev],'y30_mid_bps':y[ev],
        'constant':float(y[tr].mean()),'tree_CORE':locked.tree_CORE.to_numpy(),'tree_FLV':locked.tree_FLV.to_numpy()})
    receipt={'train_rows':len(tr),'eval_rows':len(ev),'normalization_physical_rows':len(physical),'networks':{}}
    for name,z in [('linear_snapshot',x[:,-1,:]),('linear_sequence',x.reshape(len(x),-1))]:
        beta,intercept=old.ridge_fit(z[tr].astype(float),y[tr],spec['ridge_alpha']);frame[name]=z[ev]@beta+intercept
        np.savez(stage/(prefix+'_'+name+'.npz'),beta=beta,intercept=intercept)
    for name in ('transformer','tcn'):
        frame[name],receipt['networks'][name]=fit_network(name,x,y,tr,ev,{**spec,'artifact_prefix':prefix},stage)
    frame['fold']=prefix
    frame['metric_valid']=data['valid'][ev]&(data['outcomes'][ev]<pd.Timestamp(fold['validation_end_utc_exclusive']).value)
    return frame,receipt


def pilot(spec,stage):
    qroot=Path(spec['output_root'])/'RESULTS'
    with np.load(qroot/'INPUTS.npz',allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    refs=pd.read_parquet(qroot/'FROZEN_REFERENCES.parquet');frames=[];fits={};started=time.monotonic()
    for fold in spec['folds']:
        if time.monotonic()-started>spec['pilot_max_wall_seconds']:raise RuntimeError('GC_SEQUENCE_STUDY_BUDGET')
        frame,receipt=fit_fold(spec,stage,data,refs,fold);frames.append(frame);fits[fold['fold_id']]=receipt
        io.write_json(stage/(fold['fold_id']+'_FIT_RECEIPT.json'),receipt)
        print('GC_SEQUENCE_FOLD_COMPLETE '+fold['fold_id'],flush=True)
    frame=pd.concat(frames,ignore_index=True)
    if np.any(np.diff(frame.decision_ns)<=0):raise RuntimeError('GC_SEQUENCE_POOLED_CLOCK')
    models=['constant','tree_CORE','tree_FLV','linear_snapshot','linear_sequence','transformer','tcn']
    outcome=frame.y30_mid_bps.to_numpy();valid=frame.metric_valid.to_numpy()
    metrics={n:old.p0_prediction_quality(outcome[valid],frame[n].to_numpy()[valid]) for n in models}
    by_fold={f:{n:old.p0_prediction_quality(g.y30_mid_bps.to_numpy()[g.metric_valid],g[n].to_numpy()[g.metric_valid]) for n in models} for f,g in frame.groupby('fold',sort=True)}
    for n in models:
        if not np.isfinite(frame[n]).all():raise RuntimeError('GC_SEQUENCE_NONFINITE_PREDICTION')
    if sum(4 for _ in fits)!=spec['max_fits']:raise RuntimeError('GC_SEQUENCE_FIT_BUDGET')
    frame.to_parquet(stage/'PREDICTIONS.parquet',index=False)
    smoke={'all_folds_completed':len(fits)==len(spec['folds']),'all_models_finite':True,
        'saved_network_weights_bit_exact':all(v['model_readback_bit_exact'] for f in fits.values() for v in f['networks'].values()),
        'all_networks_observed_every_train_row':all(v['unique_train_rows_seen']==f['train_rows'] for f in fits.values() for v in f['networks'].values()),
        'native_or_TEST_changed':False}
    previous=io.read_json(old.require_binding(spec['previous_recipe']));source=io.read_json(old.require_binding(previous['inputs']['native_source_plan']))
    for b in source['execution_tape_partitions']:old.require_binding(b)
    tape=old.load_tape(source['reference'],2024);d=frame.decision_ns.to_numpy();spread=old.spread_at_decision(d,tape)
    start=pd.Timestamp(spec['folds'][0]['validation_start_utc_inclusive']).value;end=pd.Timestamp(spec['folds'][-1]['validation_end_utc_exclusive']).value
    economics={};trades=[];daily=[];central_daily={}
    for name in models+['always_LONG','always_FLAT']:
        side=np.ones(len(d),np.int8) if name=='always_LONG' else np.zeros(len(d),np.int8) if name=='always_FLAT' else old.p0_actions(frame[name],spread,previous['economics'],spec['horizon_minutes'])
        economics[name],eq,ledger,info=old.p0_position_ledger(d,side,tape,start,end,previous['cost_scenarios'],previous['economics']['seconds_per_year'])
        central_daily[name]=eq[previous['central_scenario']]
        trades.extend([dict(model=name,**v) for v in ledger])
        for scenario,v in eq.items():daily.extend([{'model':name,'scenario':scenario,'date':str(day),'net_bps':float(value)} for day,value in v.items()])
    pd.DataFrame(trades).to_parquet(stage/'TRADES_AND_OPEN_MARKS.parquet',index=False)
    pd.DataFrame(daily).to_parquet(stage/'DAILY_EQUITY.parquet',index=False)
    rng=np.random.default_rng(spec['bootstrap_seed']);blocks=pd.DatetimeIndex(d,tz='UTC').strftime('%G-%V')
    groups=[np.flatnonzero(blocks==w) for w in sorted(set(blocks))];intervals={}
    alpha=spec['family_alpha']/len(spec['prediction_contrasts'])
    for candidate,reference in spec['prediction_contrasts']:
        delta=(frame[reference].to_numpy()-outcome)**2-(frame[candidate].to_numpy()-outcome)**2
        sums=np.array([delta[g][valid[g]].sum() for g in groups]);counts=np.array([valid[g].sum() for g in groups]);samples=[]
        for _ in range(spec['bootstrap_reps']):
            b=rng.integers(len(groups),size=len(groups));n=counts[b].sum()
            if n:samples.append(sums[b].sum()/n)
        intervals[candidate+'_vs_'+reference]={'MSE_improvement_bps2':float(delta[valid].mean()),
            'simultaneous_interval':np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),'week_blocks':len(groups)}
    economic_intervals={}
    for candidate,reference in spec['prediction_contrasts']:
        difference=central_daily[candidate]-central_daily[reference]
        returns=difference.diff();returns.iloc[0]=difference.iloc[0]
        week=returns.groupby(returns.index.strftime('%G-%V')).sum().to_numpy();samples=[]
        for _ in range(spec['bootstrap_reps']):samples.append(float(week[rng.integers(len(week),size=len(week))].sum()))
        economic_intervals[candidate+'_vs_'+reference]={'net_difference_bps':float(difference.iloc[-1]),
            'simultaneous_week_bootstrap_interval':np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),'week_blocks':len(week)}
    return {'status':'BOUNDED_DEVELOPMENT_PILOT_COMPLETE_NOT_NATIVE_ADMISSION','metrics':metrics,'fits':fits,'metrics_by_fold':by_fold,'smoke':smoke,
        'economics':economics,'paired_prediction_intervals':intervals,'paired_economic_intervals':economic_intervals,'models_fitted':sum(4 for _ in fits),
        'test_data_used':False,'independent_OOS':False,'automatic_expansion_authorized':False,
        'decision':'DEVELOPMENT_RESEARCH_ONLY_NO_NATIVE_OR_INDEPENDENT_EDGE_ADMISSION'}


def run(binding_path,mode):
    b=io.read_json(binding_path);io.verify_digest(PLAN,b['plan_sha256'],'SEQUENCE_PLAN');spec=io.read_json(PLAN)
    io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',b['policy_sha256'],'SEQUENCE_POLICY')
    policy=io.read_json(io.REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work'][GATE]
    if b['mode']!=mode:raise RuntimeError('GC_SEQUENCE_BINDING_MODE')
    for name,version in spec['libraries'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('GC_SEQUENCE_LIBRARY_DRIFT: '+name)
    root=Path(spec['output_root'] if mode=='qualify' else spec['pilot_output_root'])
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('GC_SEQUENCE_BINDING_PATH')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='work/gx1-current':raise RuntimeError('GC_SEQUENCE_BRANCH')
    if io.git_head()!=b['source_head'] or subprocess.check_output(['git','status','--porcelain'],text=True):raise RuntimeError('GC_SEQUENCE_FROZEN_SOURCE')
    if set(b['code_sha256'])!=set(spec['code_paths']):raise RuntimeError('GC_SEQUENCE_CODE_INVENTORY')
    for name,digest in b['code_sha256'].items():io.verify_digest(io.REPO/name,digest,'GC_SEQUENCE_CODE')
    if gate['plan_sha256']!=b['plan_sha256']:raise RuntimeError('GC_SEQUENCE_PLAN_GATE')
    if mode=='pilot':
        contract.require_pilot_authority(policy,b['plan_sha256'],b['qualification_sha256'])
        qroot=Path(spec['output_root'])/'RESULTS';io.verify_digest(qroot/'RESULT.json',b['qualification_sha256'],'QUALIFICATION')
        if b['qualification_terminal_sha256']!=gate['qualification_terminal_sha256']:raise RuntimeError('GC_SEQUENCE_QUALIFICATION_TERMINAL_BINDING')
        io.verify_digest(qroot/'TERMINAL.json',b['qualification_terminal_sha256'],'QUALIFICATION_TERMINAL')
        manifest=io.read_json(qroot/'TERMINAL.json')
        if manifest['status']!='COMPLETE' or manifest['mode']!='qualify' or manifest['exit_code']!=0 or manifest['source_unchanged'] is not True or manifest['test_data_used'] is not False:raise RuntimeError('GC_SEQUENCE_QUALIFICATION_NOT_COMPLETE')
        if set(manifest['artifacts'])!={'INPUTS.npz','COHORT.parquet','FROZEN_REFERENCES.parquet','RESULT.json','FEATURE_AUDIT.json'}:raise RuntimeError('GC_SEQUENCE_QUALIFICATION_INVENTORY')
        for name,item in manifest['artifacts'].items():io.verify_digest(qroot/name,item['sha256'],'QUALIFIED_ARTIFACT')
    elif not (gate['qualification_status']=='AUTHORIZED_SINGLE_USE' and gate['qualification_authorized'] is True
        and gate['qualification_claimed_invocations']==0 and gate['qualification_completed_invocations']==0):raise RuntimeError('GC_SEQUENCE_QUALIFICATION_CLOSED')
    if mode=='qualify' and gate['pilot_training_authorized'] is not False:raise RuntimeError('GC_SEQUENCE_QUALIFICATION_SCOPE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    stage=root/'.RESULTS.staging';stage.mkdir();start=time.monotonic()
    io.write_json(root/'EXECUTION_CLAIM.json',{'mode':mode,'binding_sha256':io.sha256_file(binding_path),'invocation':1})
    try:
        result=qualify(spec,stage) if mode=='qualify' else pilot(spec,stage)
        if time.monotonic()-start>spec['qualification_max_wall_seconds' if mode=='qualify' else 'pilot_max_wall_seconds']:raise RuntimeError('GC_SEQUENCE_TOTAL_BUDGET')
        io.write_json(stage/'RESULT.json',result)
        if mode=='qualify':
            with np.load(stage/'INPUTS.npz',allow_pickle=False) as check:
                if not np.array_equal(check['windows'][:,-1],pd.read_parquet(stage/'COHORT.parquet').physical_row):raise RuntimeError('GC_SEQUENCE_READBACK')
        artifacts={p.name:{'sha256':io.sha256_file(p),'bytes':p.stat().st_size} for p in stage.iterdir()}
        if not all(io.sha256_file(io.REPO/n)==h for n,h in b['code_sha256'].items()) or io.git_head()!=b['source_head']:raise RuntimeError('GC_SEQUENCE_SOURCE_CHANGED')
        io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','mode':mode,'exit_code':0,'source_unchanged':True,
            'test_data_used':False,'elapsed_seconds':time.monotonic()-start,'artifacts':artifacts,'capped_proof':proof,'source_head':b['source_head'],'plan_sha256':b['plan_sha256'],'binding_sha256':io.sha256_file(binding_path)})
        for p in stage.iterdir():
            with p.open('rb') as stream:os.fsync(stream.fileno())
        publish_bundle_directory_noreplace(stage,root/'RESULTS')
        print(result['status'],flush=True)
    except BaseException as exc:
        io.write_json(root/'FAILURE.json',{'error':str(exc),'mode':mode,'elapsed_seconds':time.monotonic()-start,'test_data_used':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['qualify','pilot'],required=True);p.add_argument('--binding',type=Path,required=True)
    a=p.parse_args();run(a.binding,a.mode)
