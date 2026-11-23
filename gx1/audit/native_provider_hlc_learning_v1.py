"""One explicitly admitted full-clock V2 TRAIN capacity experiment, never OOS/edge."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
core=train.core;REPO=train.REPO;POLICY=train.POLICY
E=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
NAME='NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
ANCHOR='ab6454433d269fdcd77532a4c50c036a2f45bc3c'
SELF='gx1/audit/native_provider_hlc_learning_v1.py';TEST='tests/test_native_provider_hlc_learning_v1.py';FINAL_TEST='tests/test_native_provider_hlc_final_score_v1.py';GUARD_TEST='tests/test_native_provider_hlc_learning_guard_v1.py'
BASE_PLAN={'path':str(REPO/'configs/research/NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001.json'),'sha256':'82fc015f8ea2f430cb70a8776187841e105fae14bb8ad17fd3b7afd9291b8b1b'}
REPLAY_ACCEPTANCE={'path':str(E/'PROVIDER_HLC_REPLAY_ACCEPTANCE001.json'),'sha256':'f281f8a94a68461261f34455a3aff82b0c7fa4490cc787cfb36a034995558c67'}
GUARDS=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh');ADAPTER='gx1/audit/native_future_state_provider_hlc_baseline_v1.py'
FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
EXECUTION={'class':'trainer','device':'cuda:0','memory':'20G','swap':'512M','numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128,'outer_seconds':12000,'internal_seconds':11970,'kill_after_seconds':10,'cuda_peak_reserved_bytes_max':12884901888,'GPU_core_stop_C':75,'GPU_memory_stop_C':80,'GPU_power_limit_W':250,'GPU_draw_stop_W':250,'CPU_stop_C':80,'CPU_package_stop_W':125,'host_min_available_bytes':8589934592,'host_min_commit_headroom_bytes':8589934592,'watchdog_interval_seconds':1}
SCIENTIFIC_SCOPE={'question':'Does one fixed whole-TRAIN epoch learn conditional side beyond its matched v2 prior after 194292 D1 side rows/713 target bars become genuinely observed?','change':'Only previously qualified v2 D1 provider-HLC target/prior semantics; original features/normalization/model/gradient policy/optimizer/clock order retained.','population':'All 354550 TRAIN issue rows and all five TF; unknown labels remain unknown.','primary_side_criterion':'Each pooled TF: both true sides, nonconstant logits, AUC>0.5, and strictly lower mean NLL and Brier than frozen whole-TRAIN v2 prior.','grouped_scope':'Predeclared target-start UTC years2021..2025 and D1 original/newly-known supports; report every result; no subgroup prior refit or selection.','two_sided_argmax':'Report separately; proper-score/AUC gain alone does not prove both argmax directions.','new_epochs_or_selection':False,'chronological_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'source_calendar_first_vintage_limits':'Remain open; this TRAIN diagnostic cannot qualify their historical availability.'}
def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_LEARNING_'+why)
def ref(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest()}
def read(binding):
 need(ref(binding['path'])==binding,'REFERENCE');return json.loads(Path(binding['path']).read_text())
def pub(root,name,value):
 data=value if isinstance(value,bytes)else(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();_publish_bytes(root/name,data);return ref(root/name)
def gate(binding,admitted=False):
 return {'status':'ADMITTED_EXACT_V2_WHOLE_TRAIN_ONE_EPOCH'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'remaining_invocations':int(admitted),'execution_authorized':admitted,'full_epoch_training_allowed':admitted,'additional_epochs_allowed':False,'test_authorized':False,'relaunch_allowed':False}
def validate_flags(policy,admitted):
 train.validate_training_flags(policy,main=True,admitted=admitted)
def adapter_bridge():
 before=subprocess.check_output(['git','show',ANCHOR+':'+ADAPTER],cwd=REPO);after=(REPO/ADAPTER).read_bytes();need(after.startswith(before)and len(after)>len(before),'ADAPTER_APPEND_ONLY')
 old=ast.parse(before);new=ast.parse(after);need(all(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)for a,b in zip(old.body,new.body))and len(new.body)>len(old.body),'OLD_ADAPTER_AST_UNCHANGED')
 need(all(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))for n in new.body[len(old.body):])and [n.name for n in new.body[len(old.body):]]==['score_final'],'ONLY_FINAL_SCORE_APPEND')
 return {'anchor':ANCHOR,'owner':ADAPTER,'old_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after).hexdigest(),'old_bytes_and_AST_preserved':True,'new_authority':'score_final_only'}
def build_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_learning_source_bridge
 core.future_cuda_require_runtime(runtime);need(source_commit in (None,runtime['source_commit']),'RUNTIME_SOURCE_IDENTITY')
 base=read(BASE_PLAN);need(all(runtime[k]==base['metadata']['runtime'][k]for k in ('gpu','library_versions')),'QUALIFIED_RUNTIME_COMPATIBILITY');a=read(REPLAY_ACCEPTANCE);initial=read(base['inputs']['baseline_acceptance']);main=read(base['inputs']['original_MAIN_plan'])
 need(a['cold_replay_bit_exact']is True and a['optimizer_replay_qualified']is True and len(a['matched_full_proof_components'])==10 and a['source_inputs_policy_both_boot_bookends']is True,'REPLAY_QUALIFICATION')
 need(initial['full_initialbaseline_qualified']is True and initial['actual_NN_rows']==354550 and initial['all5_TF_initial_logits_and_fullproper_scores_exact_matched_v2_prior']is True,'INITIAL_QUALIFICATION')
 recipe=copy.deepcopy(main['recipe']);need(recipe['passes']==1 and recipe['train_batch_rows']==96 and recipe['eval_batch_rows']==256 and recipe['checkpoint_every_optimizer_updates']==256,'EXACT_EXISTING_RECIPE')
 clock=copy.deepcopy(main['schedule']);clock.pop('contract_sha256');clock.pop('known_activity_by_batch');need(clock['rows']==354550 and clock['logical_batches']==3694 and clock['passes']==1 and all(clock['update_flags']),'FULL_CLOCK_SCHEDULE')
 clock.update(schema_version='gx1_provider_hlc_whole_train_schedule_v1',support_rule='All v1 update flags preserved under qualified nondecreasing v2 activity support; physically verify every batch before updating. No claimed v1 cell counts.',source_schedule_contract_sha256=main['schedule']['contract_sha256']);schedule=train.seal(clock)
 meta=copy.deepcopy(base['metadata']);meta['runtime']=runtime
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy='M5_primary');need(contract==base['model_contract'],'SAME_FORWARD_FUNCTION')
 for name,sha in base['source_owners'].items():
  if name not in (*GUARDS,ADAPTER):need(ref(REPO/name)['sha256']==sha,'UNCHANGED_NUMERIC_OWNER_'+name)
 owners=set(base['source_owners'])|{SELF,TEST,FINAL_TEST,GUARD_TEST};hashes={n:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 limits={'model_constructions':1,'optimizer_constructions':1,'feature_cache_payload_loads':1,'target_archive_loads':1,'model_forwards':5079,'backbone_forwards':5079,'readout_forwards':5079,'optimizer_steps':3694,'backward_calls':3694,'autograd_calls':0,'skipped_batches':0,'checkpoint_writes':16,'checkpoint_readbacks':16,'checkpoint_receipt_writes':16,'checkpoint_loads':0,'model_restores':0}
 return {'schema_version':'gx1_native_provider_hlc_one_epoch_plan_v1','stage':STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(owners)},'guard_source_bridge':native_provider_hlc_learning_source_bridge(hashes,producer_commit=ANCHOR),'adapter_source_bridge':adapter_bridge(),'metadata':meta,'model_contract':contract,'initial_reference':base['initial_reference'],'inputs':{**base['inputs'],'qualified_replay_plan':BASE_PLAN,'qualified_replay_acceptance':REPLAY_ACCEPTANCE},'recipe':recipe,'schedule':schedule,'backbone_gradient_policy':'M5_primary','execution':EXECUTION,'limits':limits,'output_root':str(ROOT),'scientific_scope':SCIENTIFIC_SCOPE,'authority':'ONE_SEPARATELY_ADMITTED_TRAIN_DIAGNOSTIC_EPOCH_NO_VAL_TEST_SELECTION_OR_EDGE'}
def trajectory(plan):
 return train.seal({'schema_version':'gx1_native_provider_hlc_one_epoch_trajectory_v1','inputs':plan['inputs'],'model_contract':plan['model_contract'],'schedule':plan['schedule'],'recipe':plan['recipe'],'target_contract':plan['metadata']['target_contract'],'prior_truth_sha256':plan['metadata']['truth_sha256'],'numeric_source_owners':plan['source_owners'],'runtime_compatibility':{k:plan['metadata']['runtime'][k]for k in ('gpu','library_versions')}})
def check_sources_inputs(plan):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit'],'FROZEN_CWD_BRANCH_HEAD')
 for n,digest in plan['source_owners'].items():
  path=REPO/n;need(path.is_file()and not path.is_symlink()and path.resolve()==path and ref(path)['sha256']==digest,'SOURCE_OWNER')
 for binding in plan['inputs'].values():need(ref(binding['path'])==binding,'INPUT_BYTES')
def inspect(sha,admitted):
 binding={'path':str(PLAN),'sha256':sha};p=read(binding);need(p==build_plan(p['metadata']['runtime'],p['source_commit']),'EXACT_PLAN');check_sources_inputs(p);pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[STAGE]==gate(binding,admitted),'ONE_USE_GATE');need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(p['metadata']['runtime'],live);return p,binding,pref,live

def physical(plan,binding,pref,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 began=time.monotonic();counts=dict.fromkeys(plan['limits'],0);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 batches=core.all_score_batches(np.arange(len(context['source_rows']),dtype=np.int64),plan['recipe']['train_batch_rows']);need(len(batches)==plan['schedule']['logical_batches'],'FULL_CLOCK_BATCHES')
 for j,take in enumerate(batches):need(core.digest(context['source_rows'][take])==plan['schedule']['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==plan['schedule']['batch_issue_ns_sha256'][j],'WHOLE_CLOCK_BINDING')
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 initial=plan['initial_reference'];need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'EMPTY_FRESH_OPTIMIZER');traj=trajectory(plan);steps=[];scores={'INITIAL':{'reused_qualified_result':plan['inputs']['baseline_result'],'no_repeated_initial_forwards':True}};receipts=[]
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 def checkpoint(cursor,name):
  cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane='main',cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);r=train.publish_checkpoint_receipt(ROOT/name,cp,counts,execution_ref=binding);receipts.append(r);return r
 try:
  checkpoint(0,'INITIAL.pt')
  for j,take in enumerate(batches):
   packet=adapter.packet(context,take);masks=_native_future_state_truth(packet['truth'],rows=len(take),fold_upper_ns=train.UPPER,target_contract=context['target_contract']);need(bool(masks['activity'].any())==plan['schedule']['update_flags'][j],'ACTUAL_BATCH_SUPPORT')
   steps.append(train.training_step(model,optimizer,packet,plan,counts,j));cursor=j+1;need(counts['optimizer_steps']==train.position(plan['schedule'],cursor)['optimizer_updates']and counts['skipped_batches']==0,'EXACT_LOGICAL_CURSOR');del packet,masks
   if cursor%plan['recipe']['checkpoint_every_optimizer_updates']==0 and cursor<len(batches):
    receipt=checkpoint(cursor,'UPDATE%06d.pt'%cursor);pub(ROOT,'PROGRESS_%06d.json'%cursor,{'cursor':cursor,'rows_done':min(cursor*96,354550),'rows_total':354550,'checkpoint_receipt':receipt,'last_step':steps[-1],'policy':pref,'elapsed_training_scope_seconds':time.monotonic()-began})
  optimizer.zero_grad(set_to_none=True)
  boundary=train.release_phase_allocator_cache(model,optimizer,label='TRAIN_TO_FULL_V2_FINAL_SCORE')
  scores['PASS1'],outputs=adapter.score_final(model,context,batch_rows=plan['recipe']['eval_batch_rows'],device='cuda')
  quality=quality_panels(context,outputs)
  for tf in train.TF:
   for task in ('activity','side'):
    panel=quality['all_TF'][tf][task];cell=scores['PASS1']['scores']['per_tf'][tf][task];need(panel['count']==cell['count'],'QUALITY_CANONICAL_COUNT')
    if panel['count']:
     for name,total in (('NLL','log_loss_sum'),('brier','brier_sum')):need(np.isclose(panel['model'][name]*panel['count'],cell[total],rtol=1e-12,atol=1e-7),'QUALITY_CANONICAL_SUM')
  score_ref=pub(ROOT,'FINAL_QUALITY.json',quality)
  import io
  memory=io.BytesIO();np.savez_compressed(memory,issue_ns=context['truth']['issue_ns'],**outputs);capture=pub(ROOT,'FINAL_LOGITS.npz',memory.getvalue())
  final=checkpoint(len(batches),'FINAL.pt');need(counts==plan['limits'],'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'VRAM_CAP')
  return {'counts':counts,'trajectory':traj,'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_batches':len(batches),'actual_optimizer_updates':counts['optimizer_steps'],'passes':1,'full_TRAIN_rows':354550,'whole_FINAL_score':scores['PASS1'],'final_quality':score_ref,'final_logits':capture,'final_checkpoint_receipt':final,'all_checkpoint_receipts':receipts,'allocator_boundary':boundary,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'TRAIN_side_fit_status':quality['all5_pooled_side_TRAIN_fit'],'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False}
 finally:
  for hook in hooks:hook.remove()

def run(sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_learning_execution
 began=time.monotonic()
 def stop(*_):raise TimeoutError('PROVIDER_HLC_LEARNING_SIGNAL_OR_WALL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,EXECUTION['internal_seconds'])
 try:
  guard=require_native_provider_hlc_learning_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);need(guard['plan']==binding and guard['output_root']==str(ROOT),'GUARD_PLAN_ROOT');ROOT.mkdir(exist_ok=False);identity=core.future_cpu_process_identity();claim=pub(ROOT,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'runtime':live,'process_identity':identity,'remaining_invocations':0,'TRAIN_diagnostic_only':True})
 except BaseException:
  signal.setitimer(signal.ITIMER_REAL,0);raise
 try:
  allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')};allowed|={str(ROOT/name)for name in ['INITIAL.pt','FINAL.pt','FINAL_LOGITS.npz']+['UPDATE%06d.pt'%j for j in range(256,3694,256)]}
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('LEARNING_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0])
    if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('LEARNING_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit);report=physical(plan,binding,pref,live);check_sources_inputs(plan);need(ref(PLAN)==binding and ref(POLICY)==pref,'SAME_FROZEN_PLAN_POLICY');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOTH_BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=pub(ROOT,'RESULT.json',{'status':'COMPLETE_ONE_V2_TRAIN_DIAGNOSTIC_EPOCH','plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'elapsed_seconds':time.monotonic()-began,'test_data_used':False});terminal=pub(ROOT,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False});return {'result':result,'terminal':terminal}
 except BaseException as error:pub(ROOT,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:1800],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def main():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_learning_execution
 p=argparse.ArgumentParser();p.add_argument('--plan-sha256',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--plan-sha256',a.plan_sha256]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  require_native_provider_hlc_learning_execution(plan_sha256=a.plan_sha256);plan,binding,pref,live=inspect(a.plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_NATIVE_PROVIDER_HLC_LEARNING_PASS','plan':binding,'policy':pref,'payloads_decoded':0,'models':0,'CUDA_initialized':False,'output_created':False}))
 else:print(json.dumps(run(a.plan_sha256),sort_keys=True))
# Grouped TRAIN metrics are defined before entrypoint by the integrated snippet.

def binary_panel(logits, prior_logits, labels, mask, starts):
    """Frozen-reference binary diagnostics; no refit, selected threshold or FW.

    Proper-score/equal-target algebra reuses the qualified cross-generation
    runner's binary_panel; ranking reuses train.binary_rank_metrics.
    """
    need(all(isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == len(starts)
             for a in (logits, prior_logits, labels, mask, starts))
         and labels.dtype == mask.dtype == np.bool_
         and np.isfinite(logits).all() and np.isfinite(prior_logits).all(),
         'QUALITY_PANEL_SHAPE_FINITE')
    n = int(mask.sum())
    rank = train.binary_rank_metrics(logits[mask], labels[mask])
    unique, inverse, counts = np.unique(starts[mask], return_inverse=True,
                                       return_counts=True)
    y = labels[mask].astype(np.float64)
    if n == 0:
        return {'count': 0, 'unique_target_starts': 0,
                'true_positive_count': 0, 'true_negative_count': 0,
                'true_positive_fraction': None, 'model': None,
                'matched_v2_prior': None, 'delta_model_minus_prior': None,
                'equal_target_weight': None, 'rank_and_variation': rank,
                'binary_fit_verdict': 'UNKNOWN',
                'verdict_reason': 'NO_KNOWN_OUTCOMES'}
    out = {'count': n, 'unique_target_starts': len(unique),
           'true_positive_count': int(y.sum()),
           'true_negative_count': n - int(y.sum()),
           'true_positive_fraction': float(y.mean())}
    groups, means = {}, {}
    for name, all_logits in (('model', logits), ('matched_v2_prior', prior_logits)):
        z = all_logits[mask].astype(np.float64)
        lp = -np.logaddexp(0.0, -z)
        lq = -np.logaddexp(0.0, z)
        prob = np.exp(lp)
        brier = (prob - y) ** 2
        nll = -np.where(y, lp, lq)
        by_truth, confusion = {}, {}
        for value, label in ((0, 'negative'), (1, 'positive')):
            take = y == value
            by_truth[label] = {
                'count': int(take.sum()),
                'mean_positive_probability': float(prob[take].mean()) if take.any() else None,
            }
            confusion[label] = {
                'predicted_negative': int((take & (z < 0)).sum()),
                'predicted_positive': int((take & (z > 0)).sum()),
                'exact_tie': int((take & (z == 0)).sum()),
            }
        means[name] = {
            'brier': float(brier.mean()), 'NLL': float(nll.mean()),
            'mean_positive_probability': float(prob.mean()),
            'mean_probability_minus_truth_pp': float(100 * (prob.mean() - y.mean())),
            'binary_argmax_positive_fraction': float((z > 0).mean()),
            'binary_argmax_negative_fraction': float((z < 0).mean()),
            'exact_tie_fraction': float((z == 0).mean()),
            'probability_by_true_class': by_truth,
            'strict_binary_argmax_confusion': confusion,
            'both_strict_argmax_sides_observed': bool((z < 0).any() and (z > 0).any()),
        }
        groups[name] = {
            key: float((np.bincount(inverse, weights=value, minlength=len(unique)) / counts).mean())
            for key, value in (('brier', brier), ('NLL', nll),
                               ('mean_positive_probability', prob))
        }
        out[name] = means[name]
    out['delta_model_minus_prior'] = {
        key: means['model'][key] - means['matched_v2_prior'][key]
        for key in ('brier', 'NLL')
    }
    gy = np.bincount(inverse, weights=y, minlength=len(unique)) / counts
    out['equal_target_weight'] = {
        'targets': len(unique),
        'min_known_issue_rows_per_target': int(counts.min()),
        'max_known_issue_rows_per_target': int(counts.max()),
        'targets_with_mixed_known_labels': int(((gy > 0) & (gy < 1)).sum()),
        'true_positive_fraction': float(gy.mean()), **groups,
        'delta_model_minus_prior': {
            key: groups['model'][key] - groups['matched_v2_prior'][key]
            for key in ('brier', 'NLL')
        },
        'interpretation': 'Equal weight on target means of row proper scores, not scores of mean logits; not independent/OOF.',
    }
    out['rank_and_variation'] = rank
    flags = {
        'both_true_classes': bool(rank['positive'] and rank['negative']),
        'nonconstant': rank['nonconstant'],
        'AUC_above_constant_reference': rank['AUC'] is not None and rank['AUC'] > 0.5,
        'NLL_below_frozen_prior': out['delta_model_minus_prior']['NLL'] < 0,
        'Brier_below_frozen_prior': out['delta_model_minus_prior']['brier'] < 0,
    }
    out['binary_fit_criteria'] = flags
    out['binary_fit_verdict'] = ('UNKNOWN' if not flags['both_true_classes']
                                 else 'PASS' if all(flags.values()) else 'FAIL')
    out['verdict_reason'] = ('ONE_TRUE_CLASS_AUC_UNDEFINED' if not flags['both_true_classes']
                             else 'ALL_FIXED_COMPARISONS_PASS' if all(flags.values())
                             else 'FIXED_COMPARISON_FAILED')
    return out


def quality_panels(context, outputs):
    """Score existing full-TRAIN CPU logits against one frozen v2 prior.

    Caller authenticates full context/canonical scores before this helper.
    This helper performs no IO, imports, payload decoding, fit or forward.
    """
    truth, target, prior = context['truth'], context['target_contract'], context['prior']
    fields = train.provider_hlc_truth_fields(target)
    need(all(key in truth for key in fields), 'QUALITY_TRUTH_FIELDS')
    n, width = len(truth['issue_ns']), len(train.TF)
    need(set(outputs) == {'activity_logits', 'side_logits'}
         and all(isinstance(value, np.ndarray) and value.dtype == np.float32
                 and value.shape == (n, width) and np.isfinite(value).all()
                 for value in outputs.values()), 'QUALITY_FULL_LOGITS')
    need(tuple(prior['timeframes']) == tuple(train.TF)
         and prior['target_contract_sha256'] == target['contract_sha256']
         and prior['fit_population']['issue_rows'] == n
         and prior['fit_population']['truth_sha256'] == context['truth_sha256'],
         'QUALITY_FROZEN_PRIOR_POPULATION')
    complete = truth['state_window_complete']
    partial = truth['partial_range_active']
    active = truth['activity']
    provider = truth['provider_hlc_observed']
    need(np.array_equal(truth['state_hlc_observed'], complete | provider)
         and np.array_equal(partial, active & ~complete)
         and not np.any(provider[:, :-1]), 'QUALITY_OBSERVABILITY_ALGEBRA')
    # Original target owner: market_state_native_m5_targets_v1.py:403-425.
    # V2 projection: native_future_state_provider_hlc_v1.py:158-170.
    # V1 known = active | (complete & flat_condition); side = complete & active & XOR.
    # V2 preserves active/HLC/ATR and substitutes observed=complete|provider
    # only in D1 flat/side evidence. Therefore these recover exact V1 masks
    # from authenticated retained V2 fields without loading an old archive.
    old_masks = {
        'activity': truth['activity_known'] & (complete | partial),
        'side': truth['side_known'] & complete,
    }
    masks = {'activity': truth['activity_known'], 'side': truth['side_known']}
    need(all(np.array_equal(old_masks[task][:, :-1], masks[task][:, :-1])
             for task in masks), 'QUALITY_OTHER_TF_SUPPORT_UNCHANGED')
    labels = {'activity': truth['activity'], 'side': truth['side_up']}
    years_declared = (2021, 2022, 2023, 2024, 2025)
    prior_logits = {}
    for task in masks:
        cells = [prior['per_task'][task][tf] for tf in train.TF]
        need(all(cell['fit_status'] == 'FITTED' and cell['count'] > 0 for cell in cells),
             'QUALITY_IDENTIFIED_FROZEN_PRIOR:' + task)
        probabilities = np.asarray([cell['probabilities'] for cell in cells], np.float64)
        # Exact canonical prior-logit algebra, including the float32 cast.
        prior_logits[task] = (np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])).astype(np.float32)

    def task_panels(j, take, support='all'):
        result = {}
        for task in ('activity', 'side'):
            known = masks[task][:, j]
            if support == 'v1_known':
                known = old_masks[task][:, j]
            elif support == 'newly_known':
                known = known & ~old_masks[task][:, j]
            result[task] = binary_panel(
                outputs[task + '_logits'][:, j],
                np.broadcast_to(prior_logits[task][j], (n,)),
                labels[task][:, j], take & known, truth['target_start_ns'][:, j])
        return result

    all_rows = np.ones(n, np.bool_)
    pooled, by_year, outside_year = {}, {}, {}
    years_by_tf = {}
    for j, tf in enumerate(train.TF):
        starts = truth['target_start_ns'][:, j]
        years = starts.astype('datetime64[ns]').astype('datetime64[Y]').astype(np.int64) + 1970
        years_by_tf[tf] = years
        pooled[tf] = task_panels(j, all_rows)
        by_year[tf] = {str(year): task_panels(j, years == year) for year in years_declared}
        outside = ~np.isin(years, years_declared)
        outside_year[tf] = {
            'issue_rows': int(outside.sum()),
            'target_present': int((outside & truth['target_present'][:, j]).sum()),
            'mature': int((outside & truth['mature'][:, j]).sum()),
            'immature_or_absent': int((outside & ~truth['mature'][:, j]).sum()),
            'activity_known': int((outside & masks['activity'][:, j]).sum()),
            'side_known': int((outside & masks['side'][:, j]).sum()),
            'unknown_activity': int((outside & ~masks['activity'][:, j]).sum()),
        }
    d = tuple(train.TF).index('D1')
    d1_support = {
        support: {'pooled': task_panels(d, all_rows, support),
                  'by_UTC_target_year': {
                      str(year): task_panels(d, years_by_tf['D1'] == year, support)
                      for year in years_declared}}
        for support in ('v1_known', 'newly_known')
    }
    verdicts = {tf: pooled[tf]['side']['binary_fit_verdict'] for tf in train.TF}
    aggregate = ('FAIL' if 'FAIL' in verdicts.values()
                 else 'UNKNOWN' if 'UNKNOWN' in verdicts.values() else 'PASS')
    return {
        'schema_version': 'gx1_provider_hlc_fixed_TRAIN_quality_panels_v1',
        'truth_sha256': context['truth_sha256'],
        'target_contract_sha256': target['contract_sha256'],
        'frozen_prior_contract_sha256': prior['contract_sha256'],
        'output_sha256': {key: core.digest(value) for key, value in outputs.items()},
        'rows': n, 'UTC_target_years': list(years_declared),
        'all_TF': pooled, 'by_UTC_target_year': by_year,
        'outside_declared_target_year_counts': outside_year,
        'D1_support': d1_support,
        'D1_support_reconstruction': {
            'v1_activity_known': 'v2_activity_known & (state_window_complete | partial_range_active)',
            'v1_side_known': 'v2_side_known & state_window_complete',
            'newly_known': 'v2_known & ~v1_known',
            'proof_sources': [
                'gx1/contracts/market_state_native_m5_targets_v1.py:403-425',
                'gx1/contracts/native_future_state_provider_hlc_v1.py:158-170'],
            'old_archive_loaded': False,
        },
        'all5_pooled_side_TRAIN_fit': aggregate,
        'pooled_side_verdict_by_TF': verdicts,
        'year_and_support_panels_are_diagnostics_not_forced_PASS': True,
        'prior_refitted_per_panel': False,
        'criterion': 'Both true classes, nonconstant logits, AUC>0.5, NLL and Brier strictly below the same frozen v2 whole-TRAIN prior on identical known rows.',
        'binary_argmax': 'Positive only logit>0, negative only logit<0; exact ties receive neither side.',
        'interpretation': 'Fixed whole-TRAIN fit diagnostics only. Proper-score improvement is not absolute calibration proof; argmax counts impose no class quota. Repeated TF targets are dependent. No chronological generalization, economic edge or run admission.',
    }

if __name__=='__main__':main()
