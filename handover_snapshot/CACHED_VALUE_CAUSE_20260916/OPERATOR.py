"""One bounded cache-only value-learning diagnosis; no model constructed."""
from pathlib import Path
import json, os, resource, signal, subprocess, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from gx1.contracts.unified_exit_selected_sampler_v1 import file_sha256
from gx1.contracts.unified_exit_random_access_training_v1 import frozen_policy_trace_hold_targets
from scripts.collect_gx1_handover_readonly import _native_processes

repo = Path('/home/andre2/src/GX1_CURRENT')
base = Path('/home/andre2/GX1_DATA/data/data/prebuilt/LIFECYCLE_V2_FULL_TRAIN_20260912')
obs = base/'NATIVE_REAL_TRAIN_TO_VAL_FDD70E5C/OPERATOR_OBSERVATIONS'
out = Path(__file__).parent
broad = obs/'BROAD_TRAIN95_134_20260916'
paired = base/'NATIVE_FROZEN_TRACE_MEMORY_FCD03E88_REFERENCE/LEARNING95_96_20260916'
trace = obs/'FROZEN_POLICY_TRACE_NATIVE_BATCH_20260916_V2'
start = time.monotonic()
signal.alarm(600)
torch.set_num_threads(1)
def read(p): return json.loads(p.read_text())
def git(*args): return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def require(ok,reason):
    if not ok: raise RuntimeError(reason)
bindings = {}
def bound(p, expected=None):
    h = file_sha256(p)
    require(expected is None or h == expected, 'Changed evidence '+str(p))
    bindings[str(p)] = h
    return p
def write(name,x):
    with (out/name).open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
source = git('rev-parse','HEAD')
require(source == '0dd34ff8352a3b56746a00b4a2564d128fb56ac1' and not git('status','--porcelain'), 'Source differs')
require(git('branch','--show-current') == 'work/gx1-current', 'Wrong branch')
policy = read(bound(repo/'NEXT_RUN_POLICY.json'))
require(policy['training_enabled'] is False and not _native_processes(repo), 'Training must remain stopped')
plan = {'source_commit':source,'scope':'Reuse exactly64 stored batches:1024 TRAIN Entries/4096 transitions and one16-entry five-step batch. No forward, model construction, backward, optimizer, dataset materialization, teacher refresh, VAL or TEST.',
        'questions':['Does Entry approximate side constants while its teacher provides almost no continuation?','Does the frozen teacher produce unequal effective backup lengths for LONG and SHORT?','Is observed prediction movement mainly constant rather than target-correlated?','Are early-life anchor states represented in the cached Exit training sample?','Does forecast signal reach Entry directly or only through shared representations?'],
        'criteria':'Report exact counts, conditional and centered errors/correlations, side-constant baselines and month splits. Constant baselines are descriptive same-cohort fits, not validated predictors. Reproduce stored five-step targets exactly. No causal claim from a correlation, old gradients or one batch. This measurement never opens training.'}
write('MEASUREMENT_PLAN.json',plan)
review = read(bound(repo/'handover_snapshot/FROZEN_TRACE_LEARNING95_96_REVIEW_20260916.json'))
bound(Path(review['raw_result']['path']),review['raw_result']['sha256'])
for p in ['gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py','gx1/models/entry_v10/entry_v10_ctx_train_v3.py','gx1/contracts/unified_exit_random_access_training_v1.py','gx1/contracts/unified_exit_random_access_sampler_v1.py']:
    bound(repo/p)
rows=[]; after=[]
for i in range(64):
    a=read(bound(paired/f'BATCH_{i:02d}_AFTER96.json'))
    r=read(bound(broad/f'BATCH_{i:02d}.json',a['cached_baseline']['sha256']))
    require(r['native_batch_offset'] == a['native_batch_offset'] and r['input_cache'] == a['input_cache'], 'Batch mismatch')
    rows.append(r);after.append(a['after96'])
def cat(key): return np.concatenate([np.asarray(r[key]) for r in rows],axis=0)
def pred(arm,key):
    return np.concatenate([np.asarray(a[key]) for a in after],axis=0) if arm=='AFTER96' else np.concatenate([np.asarray(r['arms']['BEFORE95'][key]) for r in rows],axis=0)
def corr(a,b):
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    return None if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
def stats(x):
    x=np.asarray(x,dtype=float).ravel()
    return {'n':int(x.size),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
def fit(p,y):
    p=np.asarray(p,float);y=np.asarray(y,float)
    return {'prediction':stats(p),'target':stats(y),'mse':float(np.mean((p-y)**2)),'mae':float(np.mean(abs(p-y))),
            'constant_target_mean_mse':float(np.var(y)),'constant_target_median_mae':float(np.mean(abs(y-np.median(y)))),
            'centered_mse':float(np.mean(((p-p.mean())-(y-y.mean()))**2)),'correlation':corr(p,y)}
def movement(p,q,y):
    d=q-p;den=float(np.mean(d*d))
    return {'change':stats(d),'constant_fraction_squared_change':float(d.mean()**2/den) if den else None,'change_target_correlation':corr(d,y),
            'before_plus_constant_mse':float(np.mean((p+d.mean()-y)**2))}
ep=cat('entry_target_bps');yp=cat('exit_target_bps');fy=cat('forecast_target_bps')
require(ep.shape==(1024,3) and yp.shape==(4096,2,2), 'Cohort differs')
require(np.all(cat('importance_weight')==1) and cat('entry_valid').all() and cat('exit_valid').all(),'Unexpected masks/weights')
require(len(set(cat('child_rows').tolist()))==1024,'Duplicate Entry rows')
months=np.array([str(np.datetime64(int(t),'ns').astype('datetime64[M]')) for t in cat('entry_time_ns')])
result={'source_commit':source,'entry':{},'exit':{},'forecast':{},'no_training_authority':True}
for s,side in enumerate(['LONG','SHORT']):
    e0=pred('BEFORE95','entry_q_bps')[:,s];e1=pred('AFTER96','entry_q_bps')[:,s]
    q0=pred('BEFORE95','exit_q_bps')[:,s,0];q1=pred('AFTER96','exit_q_bps')[:,s,0]
    result['entry'][side]={'before':fit(e0,ep[:,s]),'after':fit(e1,ep[:,s]),'movement':movement(e0,e1,ep[:,s]),
        'forecast_prediction_correlations_before':[corr(e0,pred('BEFORE95','forecast_bps')[:,h]) for h in range(4)],
        'forecast_observed_return_correlations_before':[corr(e0,fy[:,h]) for h in range(4)]}
    result['exit'][side]={'before':fit(q0,yp[:,s,0]),'after':fit(q1,yp[:,s,0]),'movement':movement(q0,q1,yp[:,s,0]),
        'teacher_bootstrap':stats(cat('bootstrap_bps')[:,s,0]),'reward':stats(cat('relative_reward_bps')[:,s,0]),
        'hold_counts':[int((q0>0).sum()),int((q1>0).sum())]}
for h,minutes in enumerate([5,25,60,120]):
    result['forecast'][str(minutes)]={}
    for arm in ['BEFORE95','AFTER96']:
        p=pred(arm,'forecast_bps')[:,h];y=fy[:,h]
        result['forecast'][str(minutes)][arm]={'fit':fit(p,y),'sign_accuracy':float(np.mean(np.sign(p)==np.sign(y))),
            'majority_sign_accuracy':float(max((y>0).mean(),(y<0).mean())),
            'months':{m:{'n':int((months==m).sum()),'correlation':corr(p[months==m],y[months==m]),'mse':float(np.mean((p[months==m]-y[months==m])**2))} for m in sorted(set(months))}}
states=[s for r in rows for s in r['states']]
ages=np.array([s['state_index'] for s in states]);entry_times=dict(zip(cat('child_rows').tolist(),cat('entry_time_ns').tolist()))
wall=np.array([(s['decision_time_ns']-entry_times[s['entry_row_index']])/60e9 for s in states])
result['sample_ages']={'observed_m1_index':stats(ages),'wall_minutes_since_entry_timestamp':stats(wall),
    'descriptive_counts_not_holding_rules':{str(x):int((ages<=x).sum()) for x in [0,5,60,1440]},
    'note':'These4096 cached training transitions do not describe all32 optimizer batches. Entry anchors are separate frozen-target queries with no Exit loss.'}
cache=torch.load(bound(trace/'BATCH_INPUTS.pt','5cff4b8ab5406f2cf68875d9e340ef617f1b877e22c56f5e54e67ae0af68e234'),map_location='cpu',weights_only=True,mmap=True)
t=torch.load(bound(trace/'TARGETS.pt','253ceaf5ec91683adcb6c4f552cba4bfb53fc2aecf9a96da4b6fa9f558f8fed8'),map_location='cpu',weights_only=True,mmap=True)
b=cache['batch'];tr=b['frozen_policy_trace'];q=t['frozen_q'][tr['successor_q_indices']];mask=b['target_action_valid_mask'][tr['successor_q_indices']]
recomputed=frozen_policy_trace_hold_targets(all_target_q=t['frozen_q'],target_action_valid_mask=b['target_action_valid_mask'],trace=tr)
require(torch.equal(recomputed,t['five_step_hold']),'Five-step cache/owner mismatch')
lengths=torch.ones((64,2),dtype=torch.long);active=torch.ones((64,2),dtype=torch.bool)
for k in range(4):
    active &= tr['transition_available_mask'][:,k+1,None] & mask[:,k,:,0] & (q[:,k,:,0]>q[:,k,:,1])
    lengths += active.long()
local=read(bound(paired/'FIVE_STEP_LOCAL_BATCH.json'))['row']
result['trace']={'exact_target_reproduction':True,'sides':{},'anchor_liquidation':stats(b['entry_liquidation_value_bps'].numpy()),
    'entry_teacher_continuation':stats(t['entry'][:,:2].numpy()-b['entry_liquidation_value_bps'].numpy()),
    'entry_one_vs_five_max_abs_difference':float(np.max(abs(t['entry'].numpy()-np.asarray(rows[0]['entry_target_bps']))))}
for s,side in enumerate(['LONG','SHORT']):
    y=t['five_step_hold'][:,s].numpy();old=t['one_step'][:,s,0].numpy();p0=np.asarray(local['arms']['BEFORE95']['exit_q_bps'])[:,s,0];p1=np.asarray(local['arms']['AFTER96']['exit_q_bps'])[:,s,0]
    result['trace']['sides'][side]={'length_counts':{str(k):int((lengths[:,s]==k).sum()) for k in range(1,6)},
        'first_successor_teacher_hold_q':stats(q[:,0,s,0].numpy()),'one_step_target':stats(old),'five_step_target':stats(y),
        'before':fit(p0,y),'after':fit(p1,y),'movement':movement(p0,p1,y),
        'available_steps':{str(k+1):int(tr['transition_available_mask'][:,k].sum()) for k in range(5)}}
gradient=read(bound(repo/'handover_snapshot/SHARED_TASK_GRADIENT_REVIEW_20260915.json'))
result['reused_gradient']={'scope':gradient['scope'],'decision':gradient['decision'],'batches':[]}
for r in gradient['rows']:
    norm=r['all_model_norms_and_support']['exit_total']['l2_norm'];g=r['groups']
    result['reused_gradient']['batches'].append({'batch':r['batch'],'head_plus_exit_blocks_fraction_squared_raw_gradient':sum(g[n]['norms_and_support']['exit_total']['l2_norm']**2 for n in ['exit_action_head','exit_blocks'])/norm**2,
        'entry_private_exit_gradient_norm':g['entry_private_encoders']['norms_and_support']['exit_total']['l2_norm'],
        'note':'Old checkpoint87 raw CPU gradients; not Adam parameter displacement or proof about checkpoint96.'})
require(not torch.cuda.is_initialized() and not _native_processes(repo), 'Unexpected GPU or training')
require(git('rev-parse','HEAD')==source and not git('status','--porcelain'),'Source changed during measurement')
result.update({'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'model_forward_count':0,'models_constructed':0,'backward_or_optimizer':False,'test_data_used':False,'gpu_used':False,
    'limits':'All rows are TRAIN; broad Exit targets are one-step diagnostics. Five-step teacher asymmetry is measured on one cached batch. Frozen-teacher near-zero value and learning collapse do not prove a market edge exists. Constant baselines are in-sample descriptions.',
    'operator_sha256':file_sha256(Path(__file__)),'input_bindings':bindings})
write('RESULT.json',result)
signal.alarm(0)
print(json.dumps({'result':str(out/'RESULT.json'),'sha256':file_sha256(out/'RESULT.json'),'elapsed_seconds':result['elapsed_seconds'],'peak_rss_kib':result['peak_rss_kib'],'decision':'CACHE_DIAGNOSIS_COMPLETE_TRAINING_REMAINS_DISABLED'}))
