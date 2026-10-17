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

PLAN=io.REPO/'configs/research/GC_SEQUENCE_QUALIFICATION_REPAIR_20261017_026.json'
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
    x=np.column_stack([f[n].to_numpy() for n in spec['fields']['F']]+
        [l[n].to_numpy() for n in spec['fields']['L']]+
        [snap[:,fields.index(n)] for n in spec['fields']['V']]+
        [base[n].to_numpy() for n in spec['fields']['basis']])
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
    return previous,d,x,c,observed,windows,tape,ex,valid,signed


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
        if not torch.isfinite(p).all():raise RuntimeError('GC_SEQUENCE_FORWARD_NONFINITE')
        output[name]={'parameters':sum(p.numel() for p in m.parameters()),'real_rows':len(x),
            'forward_seconds_single_cold_batch':time.monotonic()-start,'optimizer_steps':0,'trained':False}
    return output


def qualify(spec,stage):
    previous,d,x,c,observed,windows,tape,ex,valid,signed=load_inputs(spec)
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
    matched=predictions.set_index('decision_ns').reindex(sd[ev])
    if matched[['tree_CORE','tree_FLV','y30_mid_bps']].isna().any().any():raise RuntimeError('GC_SEQUENCE_REFERENCE_ROWS')
    if not np.array_equal(matched.y30_mid_bps.to_numpy(),ex.y_mid_bps[rows[ev]]):raise RuntimeError('GC_SEQUENCE_REFERENCE_TARGET')
    np.savez(stage/'INPUTS.npz',features=x,windows=windows,decisions=sd,outcomes=outcomes,valid=target_ok,
        y=ex.y_mid_bps[rows],train=tr,evaluation=ev,mean=mean,scale=scale,normalization_rows=unique)
    pd.DataFrame({'decision_ns':sd,'physical_row':rows,'target_valid':target_ok,'contract_index':c[rows]}).to_parquet(stage/'COHORT.parquet',index=False)
    matched.reset_index().to_parquet(stage/'FROZEN_REFERENCES.parquet',index=False)
    basis=x[rows,-1]
    return {'status':'QUALIFIED_RESEARCH_INPUTS_NOT_TRAINING_ADMISSION' if len(tr)>=spec['minimum_train_rows'] else 'QUALIFICATION_COMPLETE_INSUFFICIENT_PILOT_SUPPORT','pilot_sample_floor_met':len(tr)>=spec['minimum_train_rows'],'observed_real_data':True,
        'native':native_inventory(spec),'population_rows':len(d),'qualified_single_windows':int(observed.sum()),
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
    torch.save({'model':model.state_dict(),'target_mean':center,'target_scale':scale},stage/(name+'.pt'))
    restored=cls(x.shape[2],x.shape[1],spec[name]).eval()
    saved=torch.load(stage/(name+'.pt'),map_location='cpu',weights_only=True);restored.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():
        batch=tx[evaluation[:spec['batch_size']]]
        if not torch.equal(model(batch),restored(batch)):raise RuntimeError('GC_SEQUENCE_MODEL_READBACK')
    return pred,{'optimizer_steps':len(losses),'first_batch_loss':losses[0],'last_batch_loss':losses[-1],
        'fit_seconds':fit_seconds,'inference_seconds':inference_seconds,'inference_rows':len(evaluation),'initial_train_mse_bps2':initial_train_mse,'final_train_mse_bps2':final_train_mse,'model_readback_bit_exact':True}


def pilot(spec,stage):
    qroot=Path(spec['output_root'])/'RESULTS';data=np.load(qroot/'INPUTS.npz',allow_pickle=False)
    tr=data['train'];ev=data['evaluation']
    if len(tr)<spec['minimum_train_rows']:raise RuntimeError('GC_SEQUENCE_PILOT_COHORT_TOO_SMALL')
    x=contract.standardized(data['features'],data['windows'],data['mean'],data['scale']);y=data['y']
    refs=pd.read_parquet(qroot/'FROZEN_REFERENCES.parquet')
    frame=pd.DataFrame({'decision_ns':data['decisions'][ev],'y30_mid_bps':y[ev],
        'constant':float(y[tr].mean()),'tree_CORE':refs.tree_CORE.to_numpy(),'tree_FLV':refs.tree_FLV.to_numpy()});fits={}
    for name,z in [('linear_snapshot',x[:,-1,:]),('linear_sequence',x.reshape(len(x),-1))]:
        beta,intercept=old.ridge_fit(z[tr].astype(float),y[tr],spec['ridge_alpha']);frame[name]=z[ev]@beta+intercept
        np.savez(stage/(name+'.npz'),beta=beta,intercept=intercept)
    for name in ('transformer','tcn'):frame[name],fits[name]=fit_network(name,x,y,tr,ev,spec,stage)
    valid=data['valid'][ev];metrics={n:old.p0_prediction_quality(y[ev][valid],frame[n].to_numpy()[valid]) for n in frame.columns[2:]}
    for n in frame.columns[2:]:
        if not np.isfinite(frame[n]).all():raise RuntimeError('GC_SEQUENCE_NONFINITE_PREDICTION')
    frame.to_parquet(stage/'PREDICTIONS.parquet',index=False)
    previous=io.read_json(old.require_binding(spec['previous_recipe']));source=io.read_json(old.require_binding(previous['inputs']['native_source_plan']))
    for b in source['execution_tape_partitions']:old.require_binding(b)
    tape=old.load_tape(source['reference'],2024);d=frame.decision_ns.to_numpy();spread=old.spread_at_decision(d,tape)
    f=spec['folds'][0];start=pd.Timestamp(f['validation_start_utc_inclusive']).value;end=pd.Timestamp(f['validation_end_utc_exclusive']).value
    economics={};trades=[];daily=[];central_daily={}
    for name in list(frame.columns[2:])+['always_LONG','always_FLAT']:
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
        delta=(frame[reference].to_numpy()-y[ev])**2-(frame[candidate].to_numpy()-y[ev])**2
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
    return {'status':'BOUNDED_DEVELOPMENT_PILOT_COMPLETE_NOT_NATIVE_ADMISSION','metrics':metrics,'fits':fits,
        'economics':economics,'paired_prediction_intervals':intervals,'paired_economic_intervals':economic_intervals,'models_fitted':spec['max_fits'],
        'test_data_used':False,'independent_OOS':False,'automatic_expansion_authorized':False,
        'decision':'REVIEW_ONLY_NO_ARCHITECTURE_WINNER_FROM_SINGLE_DEVELOPMENT_FOLD'}


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
        manifest=io.read_json(qroot/'TERMINAL.json')
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
