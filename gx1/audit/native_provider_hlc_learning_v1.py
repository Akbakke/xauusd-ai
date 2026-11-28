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
 if sys.argv[1:2]==['--v3-train-eval-gap-plan-sha256']:return main_v3_train_eval_gap()
 if sys.argv[1:2]==['--v3-issue-mix-plan-sha256']:return main_v3_issue_mix()
 if sys.argv[1:2]==['--v3-block-order-recovery-plan-sha256']:return main_v3_block_order_recovery()
 if sys.argv[1:2]==['--v3-direction-block-order-plan-sha256']:return main_v3_direction_block_order()
 if sys.argv[1:2]==['--v3-direction-gradient-plan-sha256']:return main_v3_direction_gradient()
 if sys.argv[1:2]==['--v3-development-recovery-plan-sha256']:return main_v3_development_recovery()
 if sys.argv[1:2]==['--v3-development-plan-sha256']:return main_v3_development()
 if sys.argv[1:2]==['--all-tf-learning-plan-sha256']:return main_all_tf()
 if sys.argv[1:2]==['--all-tf-learning-resume-plan-sha256']:return main_all_tf_resume()
 if sys.argv[1:2]==['--resume-plan-sha256']:return main_resume()
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

# Cursor512 resume authority: additive orchestration only.
RESUME_NAME='NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004'
RESUME_STAGE=RESUME_NAME.lower();RESUME_PLAN=REPO/'configs/research'/(RESUME_NAME+'.json');RESUME_ROOT=ROOT.parent/RESUME_NAME
RESUME_ANCHOR='aa6f23bb6ad0d867fc23f3604b8399e89b633ed6'
RESUME_PRODUCER='3bd2cde102afedf81f047e9804e90765b09aad02'
RESUME_TEST='tests/test_native_provider_hlc_learning_resume_v1.py'
RESUME_GUARD_TEST='tests/test_native_provider_hlc_learning_resume_guard_v1.py'
RESUME_CURSOR=3328
RESUME_LIMITS={'model_constructions':1,'optimizer_constructions':1,'feature_cache_payload_loads':1,'target_archive_loads':1,'model_forwards':1751,'backbone_forwards':1751,'readout_forwards':1751,'optimizer_steps':366,'backward_calls':366,'autograd_calls':0,'skipped_batches':0,'checkpoint_writes':2,'checkpoint_readbacks':2,'checkpoint_receipt_writes':2,'checkpoint_loads':1,'model_restores':1}
RESUME_INPUTS={'original_plan': {'path': '/home/andre2/src/GX1_CURRENT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001.json',
                   'sha256': '68dd2bf33283ed72109e835228f964d54ebf075eb37297aba8a21d2b7826860c'},
 'original_claim': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001/CLAIM.json',
                    'sha256': '4b2e129e17757768fba79ce8cb1453edeb889ac8f32eb28893f2960b2abf9446'},
 'original_terminal': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001/TERMINAL.json',
                       'sha256': '170bf60151210a3aeb867a8359ef18b309925c39d9c778bcebe788b9139991fc'},
 'original_admitted_policy': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/POLICY_PROVIDER_HLC_LEARNING_ADMITTED001.json',
                              'sha256': '021184bfd57ce487c2987eba0ffa0bb3120d3423ca0b75d37a080fad44186f51'},
 'original_stop_review': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_LEARNING_STOP_REVIEW001.json',
                          'sha256': '35cd2828d4686da03cce9aad4e19a3b83aef0f8a18c7319d2d82ee0a54555a7b'},
 'original_backup': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_LEARNING_STOP_BACKUP001.json',
                     'sha256': 'e7a1922cd63de2604a94d09eb8ead331a1a628e4e5f723adc16da8827c65baaa'},
 'ancestral_checkpoint_receipt': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001/UPDATE000512.json',
                                  'sha256': '24ccf435e4c6d93ebb13a97e5ce613a5254d0aa58c61607f2bb27a290fd9693f'},
 'ancestral_checkpoint': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001/UPDATE000512.pt',
                          'sha256': '0b2d66011cb364da73d104be43d83951c9ff2945f836edae8f570886519563b1'},
 'ancestral_restore_proof': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_001/RESTORE512_PROOF.json',
                             'sha256': 'e47e67b774ff46747cfe36274a67e53f39fea4c7784475a93e73517eab72e14f'},
 'checkpoint_receipt': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003/UPDATE003328.json',
                        'sha256': 'eb514ac50afb53506f086a8c2dfc7d13e0ee644c2602818239af27f108d526d1'},
 'original_checkpoint': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003/UPDATE003328.pt',
                         'sha256': '9c46a0f88f1ef8f695d1319400b37c462c691e9cb150896a15bb9e8bf89a2f10'},
 'previous_execution_plan': {'path': '/home/andre2/src/GX1_CURRENT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003.json',
                             'sha256': '06a24c3d4d06e343a2cd8c2428258764a8794e68f2f71748b6c2c4cfdf0fd7f0'},
 'previous_claim': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003/CLAIM.json',
                    'sha256': '38384970d27c549d663aaa7b2f80c8628c230f4641df2d523cae1e2d454070ac'},
 'previous_terminal': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003/TERMINAL.json',
                       'sha256': '41b5955cfa7a62f52b828949eb13b65c49e908bb33738a68d2c545319a57e65c'},
 'previous_admitted_policy': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/POLICY_PROVIDER_HLC_RESUME2048_ADMITTED001.json',
                              'sha256': '24c33aefc56870ad00132e7400b7c35ab3f636ff7337c4f8ea53c56dd94cff75'},
 'previous_stop_review': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_RESUME2048_STOP_REVIEW001.json',
                          'sha256': '792b804bea5c0d3ddc5e977d67ee3c99650c46356a7205492f42f0bc370be384'},
 'previous_backup': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_RESUME2048_STOP_BACKUP001.json',
                     'sha256': '355551c64a0e0b4c017cb03fd3be123333e45287b4115c341720ad37b608cb90'},
 'previous_restore_proof': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_003/RESTORE2048_PROOF.json',
                            'sha256': '4875cb0c1e803ae48f148b7a79e1fe3467c57e546e1f501d11818639beefd619'},
 'previous_initial_receipt': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_002/UPDATE002048.json',
                              'sha256': '2813df069dc4f186e004e6dc41a53e229e921ddf8d12e8b3dd03075bdd63e474'},
 'previous_initial_checkpoint': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_002/UPDATE002048.pt',
                                 'sha256': 'cd9616fdac44e1bec311bc2ac2a6ef880b9b28ab83105376faba618d5809240e'},
 'earlier_execution_plan': {'path': '/home/andre2/src/GX1_CURRENT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_002.json',
                            'sha256': 'b16ba193e96d2d6365ab86a04a1183e5caf8b308644d9aa24e91b2cbd2529fb9'},
 'original_resume_execution_plan': {'path': '/home/andre2/src/GX1_CURRENT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_001.json',
                                    'sha256': '3dcb7d46e936f9d3b11d7166705ec7d38772d133b042d4c2cec9085d95a1bc55'},
 'intervening_checkpoint_receipt': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_001/UPDATE001792.json',
                                    'sha256': 'c7bb0d30d80ef5b27542b5cb9ab743d28aa6ce6e3c872b12d932a4e4f466b552'},
 'intervening_checkpoint': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_001/UPDATE001792.pt',
                            'sha256': 'e3eeb6853e2dfe18215d1b1817d90236ca08396a63d9b241f0faf70fa8994d78'},
 'intervening_restore_proof': {'path': '/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_002/RESTORE1792_PROOF.json',
                               'sha256': '513e2d8da5dd1bbf68aa08efb402bad33d9536933c6fdf02a75eccfc28c28ace'},
 'resume_scope': {'path': '/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/PROVIDER_HLC_RESUME3328_SCOPE001.json',
                  'sha256': '74d5c741b597c25b8aeee0977380f87a4a27e6b16a7c99f46778cf9ffc203b1b'}}


def resume_gate(binding,admitted=False,*,cursor=None):
 return {'status':('ADMITTED_EXACT_V2_CURSOR%d_REMAINING_EPOCH'%(RESUME_CURSOR if cursor is None else cursor))if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'remaining_invocations':int(admitted),'execution_authorized':admitted,'full_epoch_training_allowed':admitted,'additional_epochs_allowed':False,'test_authorized':False,'relaunch_allowed':False}


def resume_reference(binding,expected_path=None):
 need(type(binding)is dict and set(binding)in ({'path','sha256'},{'path','sha256','bytes'}),'RESUME_REFERENCE_FIELDS')
 canonical={key:binding[key]for key in ('path','sha256')}
 path=train.require_recovery_ref(canonical,expected_path=expected_path,suffix=Path(binding['path']).suffix)
 need(path.is_file()and not path.is_symlink()and path.resolve()==path and ref(path)==canonical,'RESUME_CANONICAL_REFERENCE')
 if 'bytes'in binding:need(type(binding['bytes'])is int and binding['bytes']==path.stat().st_size,'RESUME_REFERENCE_BYTES')
 return path


def resume_source_bridge():
 before=subprocess.check_output(['git','show',RESUME_ANCHOR+':'+SELF],cwd=REPO);after=(REPO/SELF).read_bytes()
 start=b'# Cursor512 resume authority: additive orchestration only.\n';end=b'# End cursor512 resume authority.\n\n'
 need(after.count(start)==after.count(end)==1,'RESUME_SINGLE_ADDITIVE_BLOCK')
 first,tail=after.split(start);_,last=tail.split(end)
 dispatch=b" if sys.argv[1:2]==['--resume-plan-sha256']:return main_resume()\n"
 need(first.count(dispatch)==1 and before==first.replace(dispatch,b'')+last,'RESUME_OLD_OWNER_BYTES_UNCHANGED_EXCEPT_DISPATCH')
 return {'anchor':RESUME_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after).hexdigest(),'old_numeric_bytes_and_AST_preserved':True,'old_main_only_change':'exact_first_token_resume_dispatch'}


def resume_original_metadata(original,inputs):
 """Authenticate the original epoch and producing execution without tensor decoding."""
 need({key:inputs[key]for key in RESUME_INPUTS}==RESUME_INPUTS,'RESUME_EXACT_ORIGINAL_REFS')
 for binding in inputs.values():resume_reference(binding)
 original_ref=inputs['original_plan'];receipt=read(inputs['checkpoint_receipt']);traj=trajectory(original)
 need(original['schema_version']=='gx1_native_provider_hlc_one_epoch_plan_v1'and original['stage']==STAGE
      and original['source_commit']==RESUME_PRODUCER and original['output_root']==str(ROOT)
      and original['execution']==EXECUTION and original['backbone_gradient_policy']=='M5_primary','RESUME_ORIGINAL_PLAN_IDENTITY')
 schedule=original['schedule'];recipe=original['recipe']
 need(schedule['rows']==354550 and schedule['logical_batches']==schedule['batches_per_pass']==3694
      and schedule['passes']==recipe['passes']==1 and schedule['optimizer_updates']==3694
      and schedule['skipped_batches']==0 and len(schedule['update_flags'])==3694
      and all(flag is True for flag in schedule['update_flags'])and recipe['train_batch_rows']==96
      and recipe['eval_batch_rows']==256 and recipe['checkpoint_every_optimizer_updates']==256,'RESUME_SAME_EPOCH_RECIPE_CLOCK')
 previous=read(inputs['previous_execution_plan'])if 'previous_execution_plan'in inputs else original
 execution_ref=inputs.get('previous_execution_plan',original_ref)
 ancestor=read(inputs['ancestral_checkpoint_receipt'])if previous is not original else receipt
 initial=read(inputs['previous_initial_receipt'])if 'previous_initial_receipt'in inputs else ancestor
 keys={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime','logical_cursor','optimizer_updates','trajectory_sha256','proof'}
 checkpoints=[(ancestor,512,original_ref,inputs.get('ancestral_checkpoint',inputs['original_checkpoint'])),(receipt,RESUME_CURSOR,execution_ref,inputs['original_checkpoint'])]
 if initial is not ancestor:checkpoints.append((initial,previous['starting_cursor'],inputs['earlier_execution_plan'],inputs['previous_initial_checkpoint']))
 if 'intervening_checkpoint_receipt'in inputs:
  intervening=read(inputs['intervening_checkpoint_receipt']);checkpoints.append((intervening,1792,inputs['original_resume_execution_plan'],inputs['intervening_checkpoint']))
 for item,cursor,execution,checkpoint_ref in checkpoints:
  need(type(item)is dict and set(item)==keys,'RESUME_RECEIPT_SCHEMA');proof=train.require_recovery_checkpoint_proof(item['proof'])
  need(item['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'
       and item['checkpoint']==checkpoint_ref and item['original_plan']==original_ref and item['execution_plan']==execution
       and item['producer_lane']=='main'and type(item['logical_cursor'])is int and type(item['optimizer_updates'])is int
       and item['logical_cursor']==item['optimizer_updates']==cursor and 0<cursor<schedule['logical_batches']
       and cursor%recipe['checkpoint_every_optimizer_updates']==0
       and item['trajectory_sha256']==traj['contract_sha256']and proof['trajectory']==core.state_sha(traj)
       and proof['position']==core.state_sha(train.position(schedule,cursor))
       and proof['scheduler_state']==core.state_sha(None)and proof['training']==core.state_sha(True)
       and proof['scores']==core.state_sha({'INITIAL':{'reused_qualified_result':original['inputs']['baseline_result'],'no_repeated_initial_forwards':True}}),'RESUME_AUTHENTIC_CURSOR_PROOFS')
 core.future_cuda_runtime_matches(original['metadata']['runtime'],ancestor['producer_runtime'])
 chains=[(original,original_ref,ancestor,'original_claim','original_terminal','original_admitted_policy',gate(original_ref,True))]
 if previous is not original:
  scope=read(inputs['resume_scope'])
  roles={'original_plan':'original_plan','checkpoint_receipt':'previous_checkpoint_receipt','original_checkpoint':'previous_checkpoint','previous_execution_plan':'previous_execution_plan','previous_claim':'previous_claim','previous_terminal':'previous_terminal','previous_admitted_policy':'previous_admitted_policy','previous_stop_review':'previous_stop_review','previous_backup':'previous_stop_backup'}
  need(all(inputs[key]==scope[role]for key,role in roles.items())and scope['starting_cursor']==RESUME_CURSOR
       and scope['new_plan_path']==str(RESUME_PLAN)and scope['new_output_root']==str(RESUME_ROOT),'RESUME_EXACT_SCOPE_REFS')
  need(previous['schema_version']=='gx1_native_provider_hlc_remaining_epoch_plan_v1'
       and previous['source_commit']==scope['previous_source_commit']
       and previous['stage']==Path(execution_ref['path']).stem.lower()
       and previous['output_root']==str(Path(receipt['checkpoint']['path']).parent)
       and previous['metadata']['original_plan']==original_ref and previous['metadata']['original']==original
       and previous['metadata']['trajectory']==traj and previous['metadata']['receipt']==initial
       and previous['inputs']['checkpoint_receipt']==inputs.get('previous_initial_receipt',inputs['ancestral_checkpoint_receipt'])
       and previous['inputs']['original_checkpoint']==inputs.get('previous_initial_checkpoint',inputs['ancestral_checkpoint'])
       and previous['starting_cursor']==initial['logical_cursor']<RESUME_CURSOR
       and previous['recipe']==recipe and previous['schedule']==schedule and previous['execution']==EXECUTION
       and previous['model_contract']==original['model_contract']and previous['initial_reference']==original['initial_reference'],'RESUME_PREVIOUS_EXECUTION_ORIGINAL_LINEAGE')
  chains.append((previous,execution_ref,receipt,'previous_claim','previous_terminal','previous_admitted_policy',resume_gate(execution_ref,True,cursor=previous['starting_cursor'])))
  if 'previous_initial_receipt'in inputs:
   need(previous['inputs']['previous_execution_plan']==inputs['earlier_execution_plan']and previous['inputs']['ancestral_checkpoint_receipt']==inputs['ancestral_checkpoint_receipt']and previous['inputs']['ancestral_checkpoint']==inputs['ancestral_checkpoint'],'RESUME_PREVIOUS_DEEP_ANCESTRY_REFS')
  proof_event=read(inputs['previous_restore_proof'])
  need(proof_event['status']=='PASS_CANONICAL_CURSOR%d_MODEL_ADAMW_RNG_RESTORE'%previous['starting_cursor']
       and proof_event['canonical_strict_restore_completed']is True and proof_event['new_model_forwards']==0
       and proof_event['original_plan']==original_ref and proof_event['execution_plan']==execution_ref
       and proof_event['checkpoint']==initial['checkpoint']and proof_event['checkpoint_receipt']==inputs.get('previous_initial_receipt',inputs['ancestral_checkpoint_receipt'])
       and proof_event['position']==train.position(schedule,previous['starting_cursor'])
       and proof_event['expected_state_proof']==initial['proof']and proof_event['uninterrupted_next_step_parity']=='NOT_MEASURED','RESUME_ANCESTRAL_RESTORE_PROOF')
  if 'ancestral_restore_proof'in inputs:
   preserved=read(inputs['ancestral_restore_proof']);need(preserved['canonical_strict_restore_completed']is True and preserved['new_model_forwards']==0 and preserved['original_plan']==original_ref and preserved['execution_plan']==inputs.get('original_resume_execution_plan',inputs['earlier_execution_plan'])and preserved['checkpoint']==ancestor['checkpoint']and preserved['checkpoint_receipt']==inputs['ancestral_checkpoint_receipt']and preserved['position']==train.position(schedule,ancestor['logical_cursor'])and preserved['expected_state_proof']==ancestor['proof'],'RESUME_PRESERVED_ORIGINAL_RESTORE_PROOF')
  if 'intervening_checkpoint_receipt'in inputs:
   need(previous['inputs']['previous_initial_receipt']==inputs['intervening_checkpoint_receipt']and previous['inputs']['previous_initial_checkpoint']==inputs['intervening_checkpoint']and previous['inputs']['earlier_execution_plan']==inputs['original_resume_execution_plan']and previous['inputs']['previous_restore_proof']==inputs['intervening_restore_proof']and previous['inputs']['ancestral_restore_proof']==inputs['ancestral_restore_proof'],'RESUME_INTERVENING_ANCESTRY_REFS')
   restored=read(inputs['intervening_restore_proof']);need(restored['canonical_strict_restore_completed']is True and restored['new_model_forwards']==0 and restored['original_plan']==original_ref and restored['execution_plan']==inputs['earlier_execution_plan']and restored['checkpoint']==intervening['checkpoint']and restored['checkpoint_receipt']==inputs['intervening_checkpoint_receipt']and restored['position']==train.position(schedule,1792)and restored['expected_state_proof']==intervening['proof'],'RESUME_INTERVENING_RESTORE_PROOF')
  review=read(inputs['previous_stop_review']);backup=read(inputs['previous_backup'])
  resume_reference(review['execution_plan'],Path(execution_ref['path']))
  need({key:review['execution_plan'][key]for key in ('path','sha256')}==execution_ref and review['original_plan']==original_ref and review['process_stopped']is True
       and review['completed_epoch']is False and review['last_durable_cursor']==RESUME_CURSOR
       and review['last_durable_rows']==RESUME_CURSOR*recipe['train_batch_rows']
       and {k:review['last_durable_checkpoint'][k]for k in ('path','sha256')}==inputs['original_checkpoint']
       and {k:review['last_durable_receipt'][k]for k in ('path','sha256')}==inputs['checkpoint_receipt']
       and {k:review['restore%d_proof'%previous['starting_cursor']][k]for k in ('path','sha256')}==inputs['previous_restore_proof']
       and review['actual_cursor%d_tensor_restore_qualified'%previous['starting_cursor']]is True and review['guard_stop_reason']=='host_maintenance_active','RESUME_PREVIOUS_STOP_REVIEW')
  expected_backup={512:('PASS_FIVE_SUFFIX_CHECKPOINTS_THROUGH1792_AND_ORIGINAL_RECEIPTS',5,43,499529425),1792:('PASS_ONE_SUFFIX_CHECKPOINT_THROUGH2048_AND_ORIGINAL_RECEIPTS',1,41,114414747),2048:('PASS_FIVE_SUFFIX_CHECKPOINTS_THROUGH3328_AND_ORIGINAL_RECEIPTS',5,40,502681793)}[previous['starting_cursor']]
  need(backup['schema_version']=='gx1_bound_separate_disk_backup_ack_v1'and backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']is True and tuple(backup[key]for key in ('status','checkpoint_files','files','total_bytes'))==expected_backup,'RESUME_PREVIOUS_SEPARATELY_SECURED')
 for execution,binding,item,claim_key,terminal_key,policy_key,expected_gate in chains:
  core.future_cuda_runtime_matches(execution['metadata']['runtime'],item['producer_runtime'])
  claim=read(inputs[claim_key]);terminal=read(inputs[terminal_key]);policy=read(inputs[policy_key])
  train.require_recovery_ref(claim['policy_at_claim'],expected_path=POLICY)
  need(claim['plan']==binding and claim['policy_at_claim']['sha256']==inputs[policy_key]['sha256']
       and claim['runtime']==item['producer_runtime']and claim['remaining_invocations']==0
       and claim['TRAIN_diagnostic_only']is True,'RESUME_FROZEN_PRODUCER_CLAIM_POLICY')
  if execution is not original:need(claim['original_plan']==original_ref and claim['starting_cursor']==execution['starting_cursor']and claim['restored_checkpoint_receipt']==inputs.get('previous_initial_receipt',inputs['ancestral_checkpoint_receipt']),'RESUME_PREVIOUS_CLAIM_LINEAGE')
  validate_flags(policy,True);need(policy[execution['stage']]==expected_gate,'RESUME_FROZEN_PRODUCER_ADMISSION')
  need(terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['plan']==binding
       and terminal['claim']==inputs[claim_key]and terminal['remaining_invocations']==0
       and terminal['relaunch_allowed']is False and 'result'not in terminal,'RESUME_FAILED_PRODUCER_TERMINAL')
  core.future_cpu_parent_stopped(claim['process_identity'])
 review=read(inputs['original_stop_review']);backup=read(inputs['original_backup'])
 need(review['plan']==original_ref and review['original_process_stopped']is True and review['completed_epoch']is False
      and review['last_durable_cursor']==ancestor['logical_cursor']and review['last_durable_rows']==ancestor['logical_cursor']*recipe['train_batch_rows']
      and {k:review['last_durable_checkpoint'][k]for k in ('path','sha256')}==ancestor['checkpoint']
      and {k:review['last_durable_receipt'][k]for k in ('path','sha256')}==inputs.get('ancestral_checkpoint_receipt',inputs['checkpoint_receipt'])
      and review['guard_stop_reason']=='host_maintenance_active','RESUME_STOP_REVIEW')
 need(backup['schema_version']=='gx1_bound_separate_disk_backup_ack_v1'
      and backup['status']=='PASS_INITIAL_AND_DURABLE256_512_CHECKPOINTS_AND_ORIGINAL_RECEIPTS'
      and backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']is True and backup['checkpoint_files']==3
      and backup['files']==32 and backup['total_bytes']==248176021,'RESUME_ORIGINALS_SEPARATELY_SECURED')
 return receipt,traj

def build_resume_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_learning_resume_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 core.future_cuda_require_runtime(runtime);need(source_commit in (None,runtime['source_commit']),'RESUME_RUNTIME_SOURCE_IDENTITY')
 original=read(RESUME_INPUTS['original_plan']);inputs={**original['inputs'],**RESUME_INPUTS}
 receipt,traj=resume_original_metadata(original,inputs)
 train.runtime_compatible(traj['runtime_compatibility'],runtime)
 meta=original['metadata'];binding=read(meta['input_binding_ref']);normalization=read(meta['normalization'])
 need(binding==meta['input_binding']and binding['input_normalization_sha256']==normalization['contract_sha256']
      and binding['feature_provider_contract_sha256']==normalization['feature_provider_contract_sha256'],'RESUME_PHYSICAL_FEATURE_NORMALIZATION_IDENTITY')
 contract=native_future_state_model_contract(binding,architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy='M5_primary')
 initial=original['initial_reference'];need(contract==original['model_contract']
      and initial['model_contract_sha256']==contract['contract_sha256']
      and initial['target_contract_sha256']==meta['target_contract']['contract_sha256']
      and initial==read(original['inputs']['baseline_initial_receipt']),'RESUME_SAME_INITIAL_FUNCTION_CONTRACT')
 for name,digest in original['source_owners'].items():
  need(hashlib.sha256(subprocess.check_output(['git','show',RESUME_PRODUCER+':'+name],cwd=REPO)).hexdigest()==digest,'RESUME_ORIGINAL_SOURCE_GIT_IDENTITY')
  if name not in (*GUARDS,SELF):need(ref(REPO/name)['sha256']==digest,'RESUME_UNCHANGED_NUMERIC_OWNER_'+name)
 bridge=resume_source_bridge();guard_hashes={name:hashlib.sha256(subprocess.check_output(['git','show',RESUME_ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in GUARDS}
 owners=set(original['source_owners'])|{RESUME_TEST,RESUME_GUARD_TEST};cursor=receipt['logical_cursor'];schedule=original['schedule'];recipe=original['recipe'];before=train.position(schedule,cursor)
 remaining=schedule['logical_batches']-cursor;updates=schedule['optimizer_updates']-before['optimizer_updates'];final_forwards=(schedule['rows']+recipe['eval_batch_rows']-1)//recipe['eval_batch_rows']
 checkpoint_cursors=list(range((cursor//recipe['checkpoint_every_optimizer_updates']+1)*recipe['checkpoint_every_optimizer_updates'],schedule['logical_batches'],recipe['checkpoint_every_optimizer_updates']))+[schedule['logical_batches']]
 limits=copy.deepcopy(original['limits']);limits.update(model_forwards=remaining+final_forwards,backbone_forwards=remaining+final_forwards,readout_forwards=remaining+final_forwards,optimizer_steps=updates,backward_calls=updates,skipped_batches=remaining-updates,checkpoint_writes=len(checkpoint_cursors),checkpoint_readbacks=len(checkpoint_cursors),checkpoint_receipt_writes=len(checkpoint_cursors),checkpoint_loads=1,model_restores=1)
 need(limits==RESUME_LIMITS,'RESUME_DERIVED_SUFFIX_COUNTS')
 return {'schema_version':'gx1_native_provider_hlc_remaining_epoch_plan_v1','stage':RESUME_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{name:ref(REPO/name)['sha256']for name in sorted(owners)},'source_bridge':bridge,'guard_source_bridge':native_provider_hlc_learning_resume_source_bridge(guard_hashes,producer_commit=RESUME_ANCHOR),'metadata':{'original':original,'original_plan':RESUME_INPUTS['original_plan'],'receipt':receipt,'trajectory':traj,'runtime':runtime},'inputs':inputs,'recipe':copy.deepcopy(original['recipe']),'schedule':copy.deepcopy(original['schedule']),'model_contract':contract,'initial_reference':copy.deepcopy(initial),'backbone_gradient_policy':'M5_primary','starting_cursor':cursor,'remaining_TRAIN_rows':schedule['rows']-min(cursor*recipe['train_batch_rows'],schedule['rows']),'remaining_FINAL_rows':schedule['rows'],'checkpoint_global_cursors':checkpoint_cursors,'execution':EXECUTION,'limits':limits,'output_root':str(RESUME_ROOT),'scientific_scope':copy.deepcopy(original['scientific_scope']),'authority':'EXACT_ORIGINAL_EPOCH_SUFFIX_FROM_DURABLE_CURSOR%d_NO_NEW_EPOCH_VAL_TEST_SELECTION_OR_EDGE'%cursor,'original_trajectory_preserved_verbatim':True,'new_INITIAL_writes':0}


def check_resume_sources_inputs(plan):
 check_sources_inputs(plan)
 for binding in plan['inputs'].values():resume_reference(binding)


def inspect_resume(sha,admitted):
 binding={'path':str(RESUME_PLAN),'sha256':sha};resume_reference(binding,RESUME_PLAN);plan=read(binding)
 need(plan==build_resume_plan(plan['metadata']['runtime'],plan['source_commit']),'RESUME_EXACT_PLAN');check_resume_sources_inputs(plan)
 pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[RESUME_STAGE]==resume_gate(binding,admitted),'RESUME_ONE_USE_GATE')
 ancestors=[(STAGE,plan['metadata']['original_plan'])]
 for role in ('original_resume_execution_plan','earlier_execution_plan','previous_execution_plan'):
  if role in plan['inputs']:ancestors.append((Path(plan['inputs'][role]['path']).stem.lower(),plan['inputs'][role]))
 for stage,ancestor in ancestors:
  old=policy[stage];need(old['plan']==ancestor and old['remaining_invocations']==0 and all(old[key]is False for key in ('execution_authorized','full_epoch_training_allowed','additional_epochs_allowed','test_authorized','relaunch_allowed')),'RESUME_CONSUMED_ANCESTOR_GATE_CLOSED')
 need(not RESUME_ROOT.exists()and not RESUME_ROOT.is_symlink(),'RESUME_FRESH_OUTPUT')
 live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 return plan,binding,pref,live


def physical_resume(plan,binding,pref,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 began=time.monotonic();meta=plan['metadata'];original=meta['original'];receipt=meta['receipt'];traj=meta['trajectory'];counts=dict.fromkeys(plan['limits'],0);start=plan['starting_cursor'];before=train.position(original['schedule'],start)
 context=adapter.load_context(original['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 batches=core.all_score_batches(np.arange(len(context['source_rows']),dtype=np.int64),original['recipe']['train_batch_rows']);need(len(batches)==original['schedule']['logical_batches'],'RESUME_FULL_CLOCK_BATCHES')
 for j,take in enumerate(batches):need(core.digest(context['source_rows'][take])==original['schedule']['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==original['schedule']['batch_issue_ns_sha256'][j],'RESUME_WHOLE_CLOCK_BINDING')
 core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 initial=original['initial_reference'];need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'RESUME_FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'RESUME_EMPTY_FRESH_OPTIMIZER')
 resume_reference(receipt['checkpoint'],Path(plan['inputs']['original_checkpoint']['path']));cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 need(cp['producer']=={'original_plan':meta['original_plan'],'lane':receipt['producer_lane'],'runtime':receipt['producer_runtime']},'RESUME_ACTUAL_CHECKPOINT_PRODUCER')
 restored=train.strict_restore(model,optimizer,cp,trajectory=traj,original_plan_ref=meta['original_plan'],expected_cursor=start,current_runtime=live,expected_state_proof=receipt['proof'])
 need(restored==train.position(original['schedule'],start),'RESUME_RESTORED_EXACT_POSITION');counts['model_restores']=1
 restore_proof=pub(RESUME_ROOT,'RESTORE%d_PROOF.json'%start,{'status':'PASS_CANONICAL_CURSOR%d_MODEL_ADAMW_RNG_RESTORE'%start,'original_plan':meta['original_plan'],'execution_plan':binding,'checkpoint':receipt['checkpoint'],'checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'position':restored,'expected_state_proof':receipt['proof'],'canonical_strict_restore_completed':True,'new_model_forwards':0,'uninterrupted_next_step_parity':'NOT_MEASURED'})
 steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp;receipts=[]
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'RESUME_FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 def checkpoint(cursor,name):
  cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=meta['original_plan'],lane='main',cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live)
  result=train.publish_checkpoint_receipt(RESUME_ROOT/name,cp,counts,execution_ref=binding);receipts.append(result);return result
 try:
  for j in range(start,len(batches)):
   take=batches[j];packet=adapter.packet(context,take);masks=_native_future_state_truth(packet['truth'],rows=len(take),fold_upper_ns=train.UPPER,target_contract=context['target_contract']);need(bool(masks['activity'].any())==original['schedule']['update_flags'][j],'RESUME_ACTUAL_BATCH_SUPPORT')
   steps.append(train.training_step(model,optimizer,packet,original,counts,j));cursor=j+1
   need(counts['optimizer_steps']+before['optimizer_updates']==train.position(original['schedule'],cursor)['optimizer_updates']and len(steps)==cursor and counts['skipped_batches']==0,'RESUME_EXACT_LOGICAL_CURSOR');del packet,masks
   if cursor%original['recipe']['checkpoint_every_optimizer_updates']==0 and cursor<len(batches):
    checkpoint_ref=checkpoint(cursor,'UPDATE%06d.pt'%cursor);pub(RESUME_ROOT,'PROGRESS_%06d.json'%cursor,{'cursor':cursor,'rows_done':min(cursor*original['recipe']['train_batch_rows'],original['schedule']['rows']),'rows_total':original['schedule']['rows'],'starting_cursor':start,'new_optimizer_updates':counts['optimizer_steps'],'checkpoint_receipt':checkpoint_ref,'last_step':steps[-1],'policy':pref,'elapsed_training_scope_seconds':time.monotonic()-began})
  optimizer.zero_grad(set_to_none=True);boundary=train.release_phase_allocator_cache(model,optimizer,label='TRAIN_TO_FULL_V2_FINAL_SCORE')
  scores['PASS1'],outputs=adapter.score_final(model,context,batch_rows=original['recipe']['eval_batch_rows'],device='cuda');quality=quality_panels(context,outputs)
  for tf in train.TF:
   for task in ('activity','side'):
    panel=quality['all_TF'][tf][task];cell=scores['PASS1']['scores']['per_tf'][tf][task];need(panel['count']==cell['count'],'RESUME_QUALITY_CANONICAL_COUNT')
    if panel['count']:
     for name,total in (('NLL','log_loss_sum'),('brier','brier_sum')):need(np.isclose(panel['model'][name]*panel['count'],cell[total],rtol=1e-12,atol=1e-7),'RESUME_QUALITY_CANONICAL_SUM')
  score_ref=pub(RESUME_ROOT,'FINAL_QUALITY.json',quality)
  import io
  memory=io.BytesIO();np.savez_compressed(memory,issue_ns=context['truth']['issue_ns'],**outputs);capture=pub(RESUME_ROOT,'FINAL_LOGITS.npz',memory.getvalue())
  final=checkpoint(len(batches),'FINAL.pt');need(counts==plan['limits'],'RESUME_EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'RESUME_VRAM_CAP')
  return {'counts':counts,'trajectory':traj,'original_plan':meta['original_plan'],'execution_plan':binding,'restored_checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'restore_proof':restore_proof,'starting_cursor':start,'remaining_TRAIN_rows':plan['remaining_TRAIN_rows'],'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_batches':len(batches),'new_optimizer_updates':counts['optimizer_steps'],'actual_optimizer_updates':before['optimizer_updates']+counts['optimizer_steps'],'passes':1,'full_TRAIN_rows':354550,'whole_FINAL_score':scores['PASS1'],'final_quality':score_ref,'final_logits':capture,'final_checkpoint_receipt':final,'all_checkpoint_receipts':receipts,'allocator_boundary':boundary,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'TRAIN_side_fit_status':quality['all5_pooled_side_TRAIN_fit'],'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False}
 finally:
  for hook in hooks:hook.remove()


def run_resume(sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_learning_resume_execution
 began=time.monotonic()
 def stop(*_):raise TimeoutError('PROVIDER_HLC_RESUME_SIGNAL_OR_WALL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,EXECUTION['internal_seconds'])
 try:
  guard=require_native_provider_hlc_learning_resume_execution(plan_sha256=sha);plan,binding,pref,live=inspect_resume(sha,True);need(guard['plan']==binding and guard['output_root']==str(RESUME_ROOT),'RESUME_GUARD_PLAN_ROOT')
  RESUME_ROOT.mkdir(exist_ok=False);identity=core.future_cpu_process_identity();claim=pub(RESUME_ROOT,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'runtime':live,'process_identity':identity,'remaining_invocations':0,'TRAIN_diagnostic_only':True,'original_plan':plan['metadata']['original_plan'],'restored_checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'starting_cursor':plan['starting_cursor']})
 except BaseException:
  signal.setitimer(signal.ITIMER_REAL,0);raise
 try:
  allowed={plan['inputs'][key]['path']for key in ('feature_archive','target_arrays','original_checkpoint','ancestral_checkpoint','previous_initial_checkpoint','intervening_checkpoint')if key in plan['inputs']};allowed|={str(RESUME_ROOT/name)for name in ['FINAL.pt','FINAL_LOGITS.npz']+['UPDATE%06d.pt'%j for j in plan['checkpoint_global_cursors']if j<plan['schedule']['logical_batches']]}
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('RESUME_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0])
    if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('RESUME_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit);report=physical_resume(plan,binding,pref,live);check_resume_sources_inputs(plan)
  need(ref(RESUME_PLAN)==binding and ref(POLICY)==pref,'RESUME_SAME_FROZEN_PLAN_POLICY')
  need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'RESUME_BOTH_BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'RESUME_M1_FIREWALL')
  result=pub(RESUME_ROOT,'RESULT.json',{'status':'COMPLETE_ORIGINAL_V2_EPOCH_FROM_CURSOR%d'%plan['starting_cursor'],'plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'elapsed_seconds':time.monotonic()-began,'test_data_used':False})
  terminal=pub(RESUME_ROOT,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  pub(RESUME_ROOT,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:1800],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)


def main_resume():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_learning_resume_execution
 p=argparse.ArgumentParser();p.add_argument('--resume-plan-sha256',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 need(sys.argv[1:]==['--resume-plan-sha256',a.resume_plan_sha256]+(['--preflight']if a.preflight else[]),'RESUME_EXACT_CLI')
 if a.preflight:
  require_native_provider_hlc_learning_resume_execution(plan_sha256=a.resume_plan_sha256);plan,binding,pref,live=inspect_resume(a.resume_plan_sha256,False)
  print(json.dumps({'status':'PREFLIGHT_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PASS','plan':binding,'policy':pref,'starting_cursor':plan['starting_cursor'],'payloads_decoded':0,'models':0,'CUDA_initialized':False,'output_created':False}))
 else:print(json.dumps(run_resume(a.resume_plan_sha256),sort_keys=True))
# End cursor512 resume authority.

# All-TF learning authority: additive orchestration only.
ALL_TF_NAME='NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002'
ALL_TF_STAGE=ALL_TF_NAME.lower();ALL_TF_PLAN=REPO/'configs/research'/(ALL_TF_NAME+'.json');ALL_TF_ROOT=ROOT.parent/ALL_TF_NAME
ALL_TF_RECOVERY_STAGE=ALL_TF_STAGE+'_recovery';ALL_TF_ANCHOR='1d83a588c21af611973afdfda6a7c960fd060436'
ALL_TF_TEST='tests/test_native_provider_hlc_all_tf_learning_v1.py';ALL_TF_GUARD_TEST='tests/test_native_provider_hlc_all_tf_learning_guard_v1.py'
ALL_TF_V2_TEST='tests/test_native_provider_hlc_all_tf_train_end_recovery_v2.py'
ALL_TF_CONTROL={'path':str(PLAN),'sha256':'68dd2bf33283ed72109e835228f964d54ebf075eb37297aba8a21d2b7826860c'}
ALL_TF_BASE={'path':str(REPO/'configs/research/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003.json'),'sha256':'540af664bb61af9fee0b445eff34bd0159a07d921db9419e1ddf573c0d1957c8'}
ALL_TF_REPLAY={'path':str(REPO/'configs/research/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001.json'),'sha256':'b0182365244c7be0b71ba2f37af34d8f65de75ccfa8ff7c3298016a377280486'}
ALL_TF_REPLAY_ACCEPTANCE={'path':str(E/'PROVIDER_HLC_ALL_TF_SEGMENTED_REPLAY_ACCEPTANCE001.json'),'sha256':'8a1ee78ec7fc3e9886458ba9bbc1feece7bb8ef9cda3307efc0b523c639c8605'}
ALL_TF_DESIGN={'path':str(E/'V2_FULL_POPULATION_NONE_COMPARISON_DESIGN001.json'),'sha256':'4e71e10200c729b11dca27c0390cb8098adf6ffe49957636681ab7cf17427a26'}
ALL_TF_PROVENANCE={'path':str(E/'V2_FULL_POPULATION_PAIRED_PROVENANCE001.json'),'sha256':'fe09eb32e8e4f6b4329c16c39053ef5989b18b087d5fbe6da812b18a5f9f52a5'}

def all_tf_gate(binding,admitted=False):
 value=gate(binding,admitted);value['status']='ADMITTED_EXACT_NONE_WHOLE_TRAIN_ONE_EPOCH'if admitted else'PREPARED_CLOSED_NOT_ADMITTED';return value

def all_tf_resume_gate(binding,admitted=False,cursor=None):
 value=all_tf_gate(binding,admitted);value.update(status='ADMITTED_EXACT_NONE_REMAINING_EPOCH'if admitted else'PREPARED_CLOSED_NOT_ADMITTED',starting_cursor=cursor,new_epoch_allowed=False);return value

def all_tf_source_bridge():
 before=subprocess.check_output(['git','show',ALL_TF_ANCHOR+':'+SELF],cwd=REPO);after=(REPO/SELF).read_bytes()
 start=b'# All-TF learning authority: additive orchestration only.\n';end=b'# End all-TF learning authority.\n\n'
 need(after.count(start)==after.count(end)==1,'ALL_TF_SINGLE_ADDITIVE_BLOCK');first,tail=after.split(start);_,last=tail.split(end)
 for dispatch in (b" if sys.argv[1:2]==['--all-tf-learning-plan-sha256']:return main_all_tf()\n",b" if sys.argv[1:2]==['--all-tf-learning-resume-plan-sha256']:return main_all_tf_resume()\n"):
  need(first.count(dispatch)==1,'ALL_TF_EXACT_MAIN_DISPATCH');first=first.replace(dispatch,b'')
 need(before==first+last,'ALL_TF_OLD_OWNER_BYTES_UNCHANGED_EXCEPT_DISPATCH')
 return {'anchor':ALL_TF_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after).hexdigest(),'old_numeric_bytes_and_AST_preserved':True,'old_main_only_change':'two_exact_first_token_all_tf_dispatches'}

def all_tf_sources(*plans):
 expected={}
 for plan in plans:expected.update(plan['source_owners'])
 for name,digest in expected.items():
  if name not in (*GUARDS,SELF):need(ref(REPO/name)['sha256']==digest,'ALL_TF_UNCHANGED_NUMERIC_OWNER_'+name)
 owners=set(expected)|{SELF,ALL_TF_TEST,ALL_TF_GUARD_TEST}
 return {name:ref(REPO/name)['sha256']for name in sorted(owners)}

def all_tf_guard_bridge():
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_all_tf_learning_source_bridge
 owners={name:hashlib.sha256(subprocess.check_output(['git','show',ALL_TF_ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in GUARDS}
 return native_provider_hlc_all_tf_learning_source_bridge(owners,producer_commit=ALL_TF_ANCHOR)

def all_tf_validate_clock(recipe,schedule):
 need(recipe['passes']==1 and recipe['train_batch_rows']==96 and recipe['eval_batch_rows']==256 and recipe['checkpoint_every_optimizer_updates']==256 and recipe['seed']==20260911 and recipe['scheduler']is None and recipe['gradient_clip_norm']==1.,'ALL_TF_EXACT_RECIPE')
 need(schedule['contract_sha256']=='ffe58ecfb95bb3e5958b31fa017db8277a56ab05c880d37c8ac383b962809631'and schedule['rows']==354550 and schedule['batch_rows']==96 and schedule['logical_batches']==schedule['batches_per_pass']==schedule['optimizer_updates']==3694 and schedule['passes']==1 and schedule['skipped_batches']==0 and schedule['shuffle']is False and schedule['drop_last']is False and len(schedule['update_flags'])==3694 and all(flag is True for flag in schedule['update_flags'])and schedule['source_rows_sha256']=='9fa3aabd86cc726c0dca3e1ac9b6629bc549f3f2524626d69075fa255f382cae'and schedule['issue_ns_sha256']=='c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea','ALL_TF_FULL_CONTROL_CLOCK')

def build_all_tf_plan(runtime,source_commit=None):
 """Bind the preregistered None comparison to its own INITIAL003 and replay."""
 core.future_cuda_require_runtime(runtime);need(source_commit in (None,runtime['source_commit']),'ALL_TF_RUNTIME_SOURCE_IDENTITY')
 control=read(ALL_TF_CONTROL);base=read(ALL_TF_BASE);replay=read(ALL_TF_REPLAY);acceptance=read(ALL_TF_REPLAY_ACCEPTANCE);design=read(ALL_TF_DESIGN);provenance=read(ALL_TF_PROVENANCE)
 initial_acceptance=read(replay['inputs']['baseline_acceptance']);initial=read(replay['inputs']['baseline_initial_receipt'])
 need(design['schema_version']=='gx1_full_population_None_comparison_design_v1'and design['TRAIN_issue_rows']==354550 and design['timeframes']==list(train.TF),'ALL_TF_PREREGISTERED_POPULATION')
 need(provenance['status']=='MATCHED_RETAINED_M5_PRIMARY_CONTROL_ELIGIBLE_FOR_NONE_TRAIN_DIAGNOSTIC'and provenance['control_plan']==ALL_TF_CONTROL and provenance['None_initial_base_plan']==ALL_TF_BASE and provenance['None_initial_receipt']==replay['inputs']['baseline_initial_receipt']and provenance['current_owner_byte_identical_to_historical']is True and provenance['new_model_forwards']==provenance['new_checkpoint_loads']==0,'ALL_TF_AUTHENTIC_PAIRED_PROVENANCE')
 need(initial_acceptance['full_initialbaseline_qualified']is True and initial_acceptance['actual_NN_forward_rows']==354550 and all(initial_acceptance['actual_reference_comparison'].values())and initial_acceptance['source_and_each_segment_bookends_pass']is True and initial_acceptance['initializer']==initial and initial_acceptance['initial_receipt']==replay['inputs']['baseline_initial_receipt']and initial_acceptance['result']==replay['inputs']['baseline_result']and replay['inputs']['baseline_plan']==ALL_TF_BASE,'ALL_TF_OWN_INITIAL_QUALIFICATION')
 need(acceptance['plan']==ALL_TF_REPLAY and acceptance['optimizer_replay_qualified']is True and acceptance['all10_proofs_bitexact']is True and len(acceptance['all10_proofs'])==10 and acceptance['both_lanes_complete_consumed']is True and acceptance['restored_nonzero_moments']['backbone']>0 and acceptance['restored_nonzero_moments']['readout']>0 and acceptance['equal_TF_eval_parity_bitexact']is True and acceptance['finite_gradient_parameters_all_bitexact']is True,'ALL_TF_OWN_REPLAY_QUALIFICATION')
 need(initial==replay['initial_reference']and initial['initial_only']is True and initial['optimizer_updates']==0 and initial['model_state_sha256']==provenance['None_model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'and initial['rng_sha256']==provenance['RNG_sha256']=='343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d','ALL_TF_OWN_INITIAL_STATE_RNG')
 meta=copy.deepcopy(base['metadata']);need(set(provenance['metadata_equal_fields'])==set(meta)-{'runtime'},'ALL_TF_COMPLETE_METADATA_PROVENANCE')
 for candidate in (control,replay):
  need(all(meta[key]==candidate['metadata'][key]for key in provenance['metadata_equal_fields']),'ALL_TF_IDENTICAL_INPUT_TRUTH_NORMALIZATION')
  need(all(runtime[key]==candidate['metadata']['runtime'][key]for key in ('gpu','library_versions')),'ALL_TF_QUALIFIED_RUNTIME_COMPATIBILITY')
 need(all(runtime[key]==meta['runtime'][key]for key in ('gpu','library_versions')),'ALL_TF_INITIAL_RUNTIME_COMPATIBILITY');meta['runtime']=copy.deepcopy(runtime)
 recipe=copy.deepcopy(control['recipe']);schedule=copy.deepcopy(control['schedule']);all_tf_validate_clock(recipe,schedule)
 need(recipe==provenance['recipe']and schedule['contract_sha256']==provenance['schedule_contract_sha256']and all(recipe[key]==replay['recipe'][key]for key in ('seed','train_batch_rows','optimizer','scheduler','gradient_clip_norm')),'ALL_TF_MATCHED_RECIPE_AND_SCHEDULE')
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy=None)
 need(base['backbone_gradient_policy']is None and replay['backbone_gradient_policy']is None and contract==base['model_contract']==replay['model_contract']and initial['model_contract_sha256']==contract['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'and initial['target_contract_sha256']==meta['target_contract']['contract_sha256'],'ALL_TF_SAME_OWN_NONE_FUNCTION')
 inputs={**copy.deepcopy(replay['inputs']),'qualified_replay_plan':ALL_TF_REPLAY,'qualified_replay_acceptance':ALL_TF_REPLAY_ACCEPTANCE,'comparison_control_plan':ALL_TF_CONTROL,'comparison_design':ALL_TF_DESIGN,'paired_provenance':ALL_TF_PROVENANCE}
 scope={'question':design['question'],'population':design['selection'],'primary_review_rule':design['primary_review_rule'],'single_intended_numerical_change':design['single_intended_numerical_change'],'new_epochs_or_selection':False,'chronological_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False,'TRAIN_diagnostic_only':True,'retained_control_not_rerun':True}
 limits=copy.deepcopy(control['limits']);need(limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==5079 and limits['optimizer_steps']==limits['backward_calls']==3694 and limits['checkpoint_writes']==limits['checkpoint_readbacks']==limits['checkpoint_receipt_writes']==16 and limits['checkpoint_loads']==limits['model_restores']==0,'ALL_TF_EXISTING_WHOLE_EPOCH_BUDGET')
 return {'schema_version':'gx1_native_provider_hlc_all_tf_one_epoch_plan_v1','stage':ALL_TF_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':all_tf_sources(control,base,replay),'source_bridge':all_tf_source_bridge(),'guard_source_bridge':all_tf_guard_bridge(),'metadata':meta,'model_contract':contract,'initial_reference':copy.deepcopy(initial),'inputs':inputs,'recipe':recipe,'schedule':schedule,'backbone_gradient_policy':None,'execution':copy.deepcopy(EXECUTION),'limits':limits,'output_root':str(ALL_TF_ROOT),'scientific_scope':scope,'authority':'ONE_SEPARATELY_ADMITTED_NONE_FULL_POPULATION_TRAIN_DIAGNOSTIC_NO_VAL_TEST_SELECTION_OR_EDGE'}

def all_tf_recovery_root(sha):
 need(type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'ALL_TF_RECOVERY_SHA256');return ALL_TF_ROOT/('RECOVERY_'+sha)

def all_tf_recovery_metadata(original,scope,*,phase_recovery=False):
 """Authenticate a stopped, consumed producer and its durable original trajectory."""
 need(type(phase_recovery)is bool,'ALL_TF_RECOVERY_PHASE_AUTHORITY')
 keys={'schema_version','original_plan','producer_plan','producer_claim','producer_terminal','producer_admitted_policy','checkpoint_receipt','checkpoint','stop_review','backup'}
 need(type(scope)is dict and set(scope)==keys and scope['schema_version']==('gx1_native_provider_hlc_all_tf_recovery_scope_v2'if phase_recovery else'gx1_native_provider_hlc_all_tf_recovery_scope_v1'),'ALL_TF_RECOVERY_SCOPE_SCHEMA')
 for key in keys-{'schema_version'}:resume_reference(scope[key])
 original_ref=scope['original_plan'];need(original_ref['path']==str(ALL_TF_PLAN)and original['schema_version']=='gx1_native_provider_hlc_all_tf_one_epoch_plan_v1'and original['stage']==ALL_TF_STAGE and original['output_root']==str(ALL_TF_ROOT)and original['backbone_gradient_policy']is None and original['execution']==EXECUTION,'ALL_TF_RECOVERY_ORIGINAL_IDENTITY')
 all_tf_validate_clock(original['recipe'],original['schedule']);control=read(ALL_TF_CONTROL)
 need(original['inputs']['comparison_control_plan']==ALL_TF_CONTROL and original['inputs']['comparison_design']==ALL_TF_DESIGN and original['inputs']['paired_provenance']==ALL_TF_PROVENANCE and original['recipe']==control['recipe']and original['schedule']==control['schedule'],'ALL_TF_RECOVERY_AUTHENTIC_ORIGINAL_CLOCK')
 initial=read(original['inputs']['baseline_initial_receipt']);need(original['initial_reference']==initial and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'and initial['model_contract_sha256']==original['model_contract']['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0','ALL_TF_RECOVERY_OWN_INITIAL_CONTRACT')
 traj=trajectory(original);receipt=read(scope['checkpoint_receipt']);proof=train.require_recovery_checkpoint_proof(receipt['proof']);cursor=receipt['logical_cursor'];producer=read(scope['producer_plan']);producer_ref=scope['producer_plan'];end=original['schedule']['logical_batches'];train_end=phase_recovery and type(cursor)is int and cursor==end
 need(set(receipt)=={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime','logical_cursor','optimizer_updates','trajectory_sha256','proof'}and receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'and receipt['checkpoint']==scope['checkpoint']and receipt['original_plan']==original_ref and receipt['execution_plan']==producer_ref and receipt['producer_lane']=='main'and type(cursor)is int and (train_end or(0<cursor<end and cursor%256==0)) and type(receipt['optimizer_updates'])is int and receipt['optimizer_updates']==cursor,'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR')
 producer_root=ALL_TF_ROOT
 if producer_ref!=original_ref:
  producer_root=all_tf_recovery_root(producer_ref['sha256']);need(producer_ref['path']==str(ALL_TF_ROOT/'RECOVERY_PLANS'/(producer_ref['sha256']+'.json'))and producer['schema_version']in(('gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1','gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2')if phase_recovery else('gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1',))and producer['stage']==ALL_TF_RECOVERY_STAGE and producer['output_root']==str(ALL_TF_ROOT)and producer['output_directory_prefix']=='RECOVERY_'and producer['metadata']['original']==original and producer['metadata']['original_plan']==original_ref and producer['metadata']['trajectory']==traj and producer['starting_cursor']<cursor and producer['model_contract']==original['model_contract']and producer['initial_reference']==original['initial_reference']and producer['backbone_gradient_policy']is None and producer['recipe']==original['recipe']and producer['schedule']==original['schedule']and producer['execution']==EXECUTION,'ALL_TF_RECOVERY_PREVIOUS_SUFFIX_LINEAGE')
 else:need(producer==original,'ALL_TF_RECOVERY_ORIGINAL_PRODUCER')
 if train_end:need(producer['schema_version']=='gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'and producer['train_end_checkpoint_before_score']is True and producer['score_only']is False,'ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER')
 checkpoint_name='TRAIN_END'if train_end else'UPDATE%06d'%cursor
 resume_reference(scope['checkpoint_receipt'],producer_root/(checkpoint_name+'.json'));resume_reference(scope['checkpoint'],producer_root/(checkpoint_name+'.pt'))
 need(receipt['trajectory_sha256']==traj['contract_sha256']and proof['trajectory']==core.state_sha(traj)and proof['position']==core.state_sha(train.position(original['schedule'],cursor))and proof['scheduler_state']==core.state_sha(None)and proof['training']==core.state_sha(True)and proof['scores']==core.state_sha({'INITIAL':{'reused_qualified_result':original['inputs']['baseline_result'],'no_repeated_initial_forwards':True}}),'ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS')
 core.future_cuda_runtime_matches(producer['metadata']['runtime'],receipt['producer_runtime']);claim=read(scope['producer_claim']);terminal=read(scope['producer_terminal']);policy=read(scope['producer_admitted_policy'])
 resume_reference(scope['producer_claim'],producer_root/'CLAIM.json');resume_reference(scope['producer_terminal'],producer_root/'TERMINAL.json');train.require_recovery_ref(claim['policy_at_claim'],expected_path=POLICY)
 need(claim['plan']==producer_ref and claim['policy_at_claim']['sha256']==scope['producer_admitted_policy']['sha256']and claim['runtime']==receipt['producer_runtime']and claim['remaining_invocations']==0 and claim['TRAIN_diagnostic_only']is True,'ALL_TF_RECOVERY_AUTHENTIC_PRODUCER_CLAIM')
 validate_flags(policy,True);expected=all_tf_gate(producer_ref,True)if producer_ref==original_ref else all_tf_resume_gate(producer_ref,True,producer['starting_cursor']);need(policy[producer['stage']]==expected,'ALL_TF_RECOVERY_FROZEN_PRODUCER_ADMISSION')
 need(terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['plan']==producer_ref and terminal['claim']==scope['producer_claim']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and 'result'not in terminal,'ALL_TF_RECOVERY_FAILED_CONSUMED_PRODUCER');core.future_cpu_parent_stopped(claim['process_identity'])
 review=read(scope['stop_review']);backup=read(scope['backup'])
 need(review['execution_plan']==producer_ref and review['original_plan']==original_ref and review['process_stopped']is True and review['completed_epoch']is False and review['last_durable_cursor']==cursor and review['last_durable_checkpoint']==scope['checkpoint']and review['last_durable_receipt']==scope['checkpoint_receipt'],'ALL_TF_RECOVERY_STOP_REVIEW')
 need(backup['schema_version']=='gx1_bound_separate_disk_backup_ack_v1'and backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']is True and backup['checkpoint']==scope['checkpoint']and backup['checkpoint_receipt']==scope['checkpoint_receipt'],'ALL_TF_RECOVERY_DURABLE_BACKUP')
 return receipt,traj,producer

def all_tf_resume_limits(original,cursor,*,phase_recovery=False):
 need(type(phase_recovery)is bool,'ALL_TF_RECOVERY_PHASE_AUTHORITY')
 if phase_recovery:need(type(cursor)is int and 0<cursor<=original['schedule']['logical_batches']and(cursor==original['schedule']['logical_batches']or cursor%256==0),'ALL_TF_RECOVERY_PHASE_CURSOR')
 schedule=original['schedule'];recipe=original['recipe'];before=train.position(schedule,cursor);remaining=schedule['logical_batches']-cursor;updates=schedule['optimizer_updates']-before['optimizer_updates'];final_forwards=(schedule['rows']+recipe['eval_batch_rows']-1)//recipe['eval_batch_rows']
 cadence=recipe['checkpoint_every_optimizer_updates'];cursors=list(range((cursor//cadence+1)*cadence,schedule['logical_batches'],cadence))+[schedule['logical_batches']]
 checkpoint_count=len(cursors)+int(phase_recovery and remaining>0)
 limits=copy.deepcopy(original['limits']);limits.update(model_forwards=remaining+final_forwards,backbone_forwards=remaining+final_forwards,readout_forwards=remaining+final_forwards,optimizer_steps=updates,backward_calls=updates,skipped_batches=remaining-updates,checkpoint_writes=checkpoint_count,checkpoint_readbacks=checkpoint_count,checkpoint_receipt_writes=checkpoint_count,checkpoint_loads=1,model_restores=1);return limits,cursors

def build_all_tf_resume_plan(runtime,scope,source_commit=None,*,phase_recovery=False):
 need(type(phase_recovery)is bool,'ALL_TF_RECOVERY_PHASE_AUTHORITY')
 core.future_cuda_require_runtime(runtime);need(source_commit in (None,runtime['source_commit']),'ALL_TF_RECOVERY_RUNTIME_SOURCE_IDENTITY');resume_reference(scope);scope_value=read(scope);original=read(scope_value['original_plan']);receipt,traj,producer=all_tf_recovery_metadata(original,scope_value,phase_recovery=phase_recovery);train.runtime_compatible(traj['runtime_compatibility'],runtime)
 meta=original['metadata'];binding=read(meta['input_binding_ref']);normalization=read(meta['normalization']);need(binding==meta['input_binding']and binding['input_normalization_sha256']==normalization['contract_sha256']and binding['feature_provider_contract_sha256']==normalization['feature_provider_contract_sha256'],'ALL_TF_RECOVERY_PHYSICAL_INPUT_IDENTITY')
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 contract=native_future_state_model_contract(binding,architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy=None);need(contract==original['model_contract'],'ALL_TF_RECOVERY_NONE_FUNCTION')
 for plan in (original,producer):
  for name,digest in plan['source_owners'].items():need(hashlib.sha256(subprocess.check_output(['git','show',plan['source_commit']+':'+name],cwd=REPO)).hexdigest()==digest,'ALL_TF_RECOVERY_PRODUCER_GIT_IDENTITY')
 cursor=receipt['logical_cursor'];limits,cursors=all_tf_resume_limits(original,cursor,phase_recovery=phase_recovery);inputs={**original['inputs'],**{key:value for key,value in scope_value.items()if key!='schema_version'},'recovery_scope':copy.deepcopy(scope)}
 owners=all_tf_sources(original,producer)
 if phase_recovery:owners[ALL_TF_V2_TEST]=ref(REPO/ALL_TF_V2_TEST)['sha256']
 plan={'schema_version':('gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'if phase_recovery else'gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1'),'stage':ALL_TF_RECOVERY_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':owners,'source_bridge':all_tf_source_bridge(),'guard_source_bridge':all_tf_guard_bridge(),'metadata':{'original':original,'original_plan':scope_value['original_plan'],'receipt':receipt,'trajectory':traj,'runtime':copy.deepcopy(runtime)},'inputs':inputs,'recipe':copy.deepcopy(original['recipe']),'schedule':copy.deepcopy(original['schedule']),'model_contract':contract,'initial_reference':copy.deepcopy(original['initial_reference']),'backbone_gradient_policy':None,'starting_cursor':cursor,'remaining_TRAIN_rows':original['schedule']['rows']-min(cursor*original['recipe']['train_batch_rows'],original['schedule']['rows']),'remaining_FINAL_rows':original['schedule']['rows'],'checkpoint_global_cursors':cursors,'execution':copy.deepcopy(EXECUTION),'limits':limits,'output_root':str(ALL_TF_ROOT),'output_directory_prefix':'RECOVERY_','scientific_scope':copy.deepcopy(original['scientific_scope']),'authority':'EXACT_ORIGINAL_NONE_EPOCH_SUFFIX_NO_NEW_EPOCH_VAL_TEST_SELECTION_OR_EDGE','original_trajectory_preserved_verbatim':True,'new_INITIAL_writes':0}
 if phase_recovery:plan.update(train_end_checkpoint_before_score=True,score_only=cursor==original['schedule']['logical_batches'])
 return plan

def build_all_tf_resume_plan_v2(runtime,scope,source_commit=None):
 """Explicit phase recovery: retain trained cursor3694 before full scoring."""
 return build_all_tf_resume_plan(runtime,scope,source_commit,phase_recovery=True)

def inspect_all_tf(sha,admitted):
 binding={'path':str(ALL_TF_PLAN),'sha256':sha};plan=read(binding);need(plan==build_all_tf_plan(plan['metadata']['runtime'],plan['source_commit']),'ALL_TF_EXACT_PLAN');check_sources_inputs(plan);pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[ALL_TF_STAGE]==all_tf_gate(binding,admitted),'ALL_TF_ONE_USE_GATE');need(not ALL_TF_ROOT.exists()and not ALL_TF_ROOT.is_symlink(),'ALL_TF_FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def inspect_all_tf_resume(sha,admitted):
 output=all_tf_recovery_root(sha);path=ALL_TF_ROOT/'RECOVERY_PLANS'/(sha+'.json');binding={'path':str(path),'sha256':sha};resume_reference(binding,path);plan=read(binding)
 builder=build_all_tf_resume_plan_v2 if plan['schema_version']=='gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'else build_all_tf_resume_plan
 need(plan==builder(plan['metadata']['runtime'],plan['inputs']['recovery_scope'],plan['source_commit']),'ALL_TF_RECOVERY_EXACT_PLAN');check_sources_inputs(plan);pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[ALL_TF_RECOVERY_STAGE]==all_tf_resume_gate(binding,admitted,plan['starting_cursor']),'ALL_TF_RECOVERY_ONE_USE_GATE')
 original_gate=policy[ALL_TF_STAGE];need(original_gate['plan']==plan['metadata']['original_plan']and original_gate['remaining_invocations']==0 and all(original_gate[key]is False for key in ('execution_authorized','full_epoch_training_allowed','additional_epochs_allowed','test_authorized','relaunch_allowed')),'ALL_TF_RECOVERY_ORIGINAL_GATE_CONSUMED')
 need(not output.exists()and not output.is_symlink(),'ALL_TF_RECOVERY_FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def physical_all_tf(plan,binding,pref,live,*,order_scope=None):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_SCOPE
 order_root=ALL_TF_ROOT;canonical_schedule=plan['schedule']
 if order_scope is not None:
  from gx1.audit import native_provider_hlc_block_order_v1 as order
# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 MAIN_MODULE
  if type(order_scope)is dict and order_scope.get('module')=='gx1.audit.native_provider_hlc_row_membership_fit_v1':
   from gx1.audit import native_provider_hlc_row_membership_fit_v1 as order
# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 MAIN_MODULE
  order_root=order.require_engine_scope(order_scope,plan,binding,'main');canonical_schedule=plan['schedule']['canonical_schedule']
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_SCOPE
 began=time.monotonic();counts=dict.fromkeys(plan['limits'],0);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 batches=core.all_score_batches(np.arange(len(context['source_rows']),dtype=np.int64),plan['recipe']['train_batch_rows']);need(len(batches)==plan['schedule']['logical_batches'],'FULL_CLOCK_BATCHES')
 for j,take in enumerate(batches):need(core.digest(context['source_rows'][take])==canonical_schedule['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==canonical_schedule['batch_issue_ns_sha256'][j],'WHOLE_CLOCK_BINDING')
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_BATCHES
# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 MAIN_BATCHES
 if order_scope is not None:
  if order_scope.get('module')=='gx1.audit.native_provider_hlc_row_membership_fit_v1':batches=order.resolve_batches(context,plan['schedule'],counts)
  else:batches=order.resolve_batches(context,plan['schedule'])
# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 MAIN_BATCHES
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_BATCHES
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=plan['initial_reference'];need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'EMPTY_FRESH_OPTIMIZER');traj=trajectory(plan);steps=[];scores={'INITIAL':{'reused_qualified_result':plan['inputs']['baseline_result'],'no_repeated_initial_forwards':True}};receipts=[]
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 def checkpoint(cursor,name):
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_CHECKPOINT
  if order_scope is not None:order.validate_prefix(plan['schedule'],steps,cursor)
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_CHECKPOINT
  cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane='main',cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);r=train.publish_checkpoint_receipt(order_root/name,cp,counts,execution_ref=binding);receipts.append(r);return r
 try:
  checkpoint(0,'INITIAL.pt')
  for j,take in enumerate(batches):
   packet=adapter.packet(context,take);masks=_native_future_state_truth(packet['truth'],rows=len(take),fold_upper_ns=train.UPPER,target_contract=context['target_contract']);need(bool(masks['activity'].any())==plan['schedule']['update_flags'][j],'ACTUAL_BATCH_SUPPORT')
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_PACKET
   if order_scope is not None:order_record=order.step_binding(context,plan['schedule'],j,take,packet)
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_PACKET
   steps.append(train.training_step(model,optimizer,packet,plan,counts,j));cursor=j+1;need(counts['optimizer_steps']==train.position(plan['schedule'],cursor)['optimizer_updates']and counts['skipped_batches']==0,'EXACT_LOGICAL_CURSOR');del packet,masks
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_JOURNAL
   if order_scope is not None:steps[-1]['order_binding']=order_record
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_JOURNAL
   if cursor%plan['recipe']['checkpoint_every_optimizer_updates']==0 and cursor<len(batches):
    receipt=checkpoint(cursor,'UPDATE%06d.pt'%cursor);pub(order_root,'PROGRESS_%06d.json'%cursor,{'cursor':cursor,'rows_done':min(cursor*96,354550),'rows_total':354550,'checkpoint_receipt':receipt,'last_step':steps[-1],'policy':pref,'elapsed_training_scope_seconds':time.monotonic()-began})
  optimizer.zero_grad(set_to_none=True)
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_TRAIN_END
  if order_scope is not None:
   order.validate_prefix(plan['schedule'],steps,len(batches))
   need(model.training is True and scores=={'INITIAL':{'reused_qualified_result':plan['inputs']['baseline_result'],'no_repeated_initial_forwards':True}},'ORDER_TRAIN_END_INITIAL_ONLY')
   train_end_receipt=checkpoint(len(batches),'TRAIN_END.pt')
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_TRAIN_END
  boundary=train.release_phase_allocator_cache(model,optimizer,label='TRAIN_TO_FULL_V2_FINAL_SCORE')
  scores['PASS1'],outputs=adapter.score_final(model,context,batch_rows=plan['recipe']['eval_batch_rows'],device='cuda')
  quality=quality_panels(context,outputs)
  for tf in train.TF:
   for task in ('activity','side'):
    panel=quality['all_TF'][tf][task];cell=scores['PASS1']['scores']['per_tf'][tf][task];need(panel['count']==cell['count'],'QUALITY_CANONICAL_COUNT')
    if panel['count']:
     for name,total in (('NLL','log_loss_sum'),('brier','brier_sum')):need(np.isclose(panel['model'][name]*panel['count'],cell[total],rtol=1e-12,atol=1e-7),'QUALITY_CANONICAL_SUM')
  score_ref=pub(order_root,'FINAL_QUALITY.json',quality)
  import io
  memory=io.BytesIO();np.savez_compressed(memory,issue_ns=context['truth']['issue_ns'],**outputs);capture=pub(order_root,'FINAL_LOGITS.npz',memory.getvalue())
  final=checkpoint(len(batches),'FINAL.pt');need(counts==plan['limits'],'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'VRAM_CAP')
  report={'counts':counts,'trajectory':traj,'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_batches':len(batches),'actual_optimizer_updates':counts['optimizer_steps'],'passes':1,'full_TRAIN_rows':354550,'whole_FINAL_score':scores['PASS1'],'final_quality':score_ref,'final_logits':capture,'final_checkpoint_receipt':final,'all_checkpoint_receipts':receipts,'allocator_boundary':boundary,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'TRAIN_side_fit_status':quality['all5_pooled_side_TRAIN_fit'],'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False}
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_REPORT
  if order_scope is not None:report.update(train_end_checkpoint_receipt=train_end_receipt,order_cursor=order.cursor_binding(plan['schedule'],len(batches)),actual_order_journal_verified=True)
  return report
# END GX1 BLOCK ORDER ENGINE 20261124 001 MAIN_REPORT
 finally:
  for hook in hooks:hook.remove()

def physical_all_tf_resume(plan,binding,pref,live,*,order_scope=None):
 output=all_tf_recovery_root(binding['sha256']);phase_recovery=plan['schema_version']=='gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 began=time.monotonic();meta=plan['metadata'];original=meta['original'];receipt=meta['receipt'];traj=meta['trajectory'];counts=dict.fromkeys(plan['limits'],0);start=plan['starting_cursor'];before=train.position(original['schedule'],start)
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_SCOPE
 canonical_schedule=original['schedule']
 if order_scope is not None:
  from gx1.audit import native_provider_hlc_block_order_v1 as order
# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 RESUME_MODULE
  if type(order_scope)is dict and order_scope.get('module')=='gx1.audit.native_provider_hlc_row_membership_fit_v1':
   from gx1.audit import native_provider_hlc_row_membership_fit_v1 as order
# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 RESUME_MODULE
  output=order.require_engine_scope(order_scope,plan,binding,'resume');phase_recovery=True;canonical_schedule=original['schedule']['canonical_schedule']
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_SCOPE
 context=adapter.load_context(original['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 batches=core.all_score_batches(np.arange(len(context['source_rows']),dtype=np.int64),original['recipe']['train_batch_rows']);need(len(batches)==original['schedule']['logical_batches'],'RESUME_FULL_CLOCK_BATCHES')
 for j,take in enumerate(batches):need(core.digest(context['source_rows'][take])==canonical_schedule['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==canonical_schedule['batch_issue_ns_sha256'][j],'RESUME_WHOLE_CLOCK_BINDING')
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_BATCHES
# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 RESUME_BATCHES
 if order_scope is not None:
  if order_scope.get('module')=='gx1.audit.native_provider_hlc_row_membership_fit_v1':batches=order.resolve_batches(context,original['schedule'],counts)
  else:batches=order.resolve_batches(context,original['schedule'])
# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 RESUME_BATCHES
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_BATCHES
 core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=original['initial_reference'];need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'RESUME_FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'RESUME_EMPTY_FRESH_OPTIMIZER')
 resume_reference(receipt['checkpoint'],Path(plan['inputs']['checkpoint']['path']));cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 need(cp['producer']=={'original_plan':meta['original_plan'],'lane':receipt['producer_lane'],'runtime':receipt['producer_runtime']},'RESUME_ACTUAL_CHECKPOINT_PRODUCER')
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_PREFIX
 if order_scope is not None:
  order.validate_prefix(original['schedule'],cp['steps'],start);factory_buffers=core.state_sha(dict(model.named_buffers()))
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_PREFIX
 restored=train.strict_restore(model,optimizer,cp,trajectory=traj,original_plan_ref=meta['original_plan'],expected_cursor=start,current_runtime=live,expected_state_proof=receipt['proof'])
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_BUFFERS
 if order_scope is not None:need(core.state_sha(dict(model.named_buffers()))==factory_buffers,'ORDER_RESTORE_FACTORY_BUFFERS')
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_BUFFERS
 need(restored==train.position(original['schedule'],start),'RESUME_RESTORED_EXACT_POSITION');counts['model_restores']=1
 restore_proof=pub(output,'RESTORE%d_PROOF.json'%start,{'status':'PASS_CANONICAL_CURSOR%d_MODEL_ADAMW_RNG_RESTORE'%start,'original_plan':meta['original_plan'],'execution_plan':binding,'checkpoint':receipt['checkpoint'],'checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'position':restored,'expected_state_proof':receipt['proof'],'canonical_strict_restore_completed':True,'new_model_forwards':0,'uninterrupted_next_step_parity':'NOT_MEASURED'})
 steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp;receipts=[]
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'RESUME_FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 def checkpoint(cursor,name):
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_CHECKPOINT
  if order_scope is not None:order.validate_prefix(original['schedule'],steps,cursor)
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_CHECKPOINT
  cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=meta['original_plan'],lane='main',cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live)
  result=train.publish_checkpoint_receipt(output/name,cp,counts,execution_ref=binding);receipts.append(result);return result
 try:
  for j in range(start,len(batches)):
   take=batches[j];packet=adapter.packet(context,take);masks=_native_future_state_truth(packet['truth'],rows=len(take),fold_upper_ns=train.UPPER,target_contract=context['target_contract']);need(bool(masks['activity'].any())==original['schedule']['update_flags'][j],'RESUME_ACTUAL_BATCH_SUPPORT')
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_PACKET
   if order_scope is not None:order_record=order.step_binding(context,original['schedule'],j,take,packet)
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_PACKET
   steps.append(train.training_step(model,optimizer,packet,original,counts,j));cursor=j+1
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_JOURNAL
   if order_scope is not None:steps[-1]['order_binding']=order_record
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_JOURNAL
   need(counts['optimizer_steps']+before['optimizer_updates']==train.position(original['schedule'],cursor)['optimizer_updates']and len(steps)==cursor and counts['skipped_batches']==0,'RESUME_EXACT_LOGICAL_CURSOR');del packet,masks
   if cursor%original['recipe']['checkpoint_every_optimizer_updates']==0 and cursor<len(batches):
    checkpoint_ref=checkpoint(cursor,'UPDATE%06d.pt'%cursor);pub(output,'PROGRESS_%06d.json'%cursor,{'cursor':cursor,'rows_done':min(cursor*original['recipe']['train_batch_rows'],original['schedule']['rows']),'rows_total':original['schedule']['rows'],'starting_cursor':start,'new_optimizer_updates':counts['optimizer_steps'],'checkpoint_receipt':checkpoint_ref,'last_step':steps[-1],'policy':pref,'elapsed_training_scope_seconds':time.monotonic()-began})
  optimizer.zero_grad(set_to_none=True)
  if phase_recovery:
   need(model.training is True and len(steps)==len(batches)and scores=={'INITIAL':{'reused_qualified_result':original['inputs']['baseline_result'],'no_repeated_initial_forwards':True}},'TRAIN_END_EXACT_TRAINING_STEPS_INITIAL_ONLY_SCORES')
   train_end_receipt=checkpoint(len(batches),'TRAIN_END.pt')if start<len(batches)else plan['inputs']['checkpoint_receipt']
  boundary=train.release_phase_allocator_cache(model,optimizer,label='TRAIN_TO_FULL_V2_FINAL_SCORE')
  scores['PASS1'],outputs=adapter.score_final(model,context,batch_rows=original['recipe']['eval_batch_rows'],device='cuda');quality=quality_panels(context,outputs)
  for tf in train.TF:
   for task in ('activity','side'):
    panel=quality['all_TF'][tf][task];cell=scores['PASS1']['scores']['per_tf'][tf][task];need(panel['count']==cell['count'],'RESUME_QUALITY_CANONICAL_COUNT')
    if panel['count']:
     for name,total in (('NLL','log_loss_sum'),('brier','brier_sum')):need(np.isclose(panel['model'][name]*panel['count'],cell[total],rtol=1e-12,atol=1e-7),'RESUME_QUALITY_CANONICAL_SUM')
  score_ref=pub(output,'FINAL_QUALITY.json',quality)
  import io
  memory=io.BytesIO();np.savez_compressed(memory,issue_ns=context['truth']['issue_ns'],**outputs);capture=pub(output,'FINAL_LOGITS.npz',memory.getvalue())
  final=checkpoint(len(batches),'FINAL.pt');need(counts==plan['limits'],'RESUME_EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'RESUME_VRAM_CAP')
  report={'counts':counts,'trajectory':traj,'original_plan':meta['original_plan'],'execution_plan':binding,'restored_checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'restore_proof':restore_proof,'starting_cursor':start,'remaining_TRAIN_rows':plan['remaining_TRAIN_rows'],'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_batches':len(batches),'new_optimizer_updates':counts['optimizer_steps'],'actual_optimizer_updates':before['optimizer_updates']+counts['optimizer_steps'],'passes':1,'full_TRAIN_rows':354550,'whole_FINAL_score':scores['PASS1'],'final_quality':score_ref,'final_logits':capture,'final_checkpoint_receipt':final,'all_checkpoint_receipts':receipts,'allocator_boundary':boundary,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'TRAIN_side_fit_status':quality['all5_pooled_side_TRAIN_fit'],'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False}
  if phase_recovery:report.update(train_end_checkpoint_receipt=train_end_receipt,score_only=start==len(batches))
# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_REPORT
  if order_scope is not None:report.update(order_cursor=order.cursor_binding(original['schedule'],len(batches)),actual_order_journal_verified=True)
# END GX1 BLOCK ORDER ENGINE 20261124 001 RESUME_REPORT
  return report
 finally:
  for hook in hooks:hook.remove()

def run_all_tf(sha,mode='main'):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_learning_execution
 need(mode in ('main','resume'),'ALL_TF_EXACT_MODE');output=ALL_TF_ROOT if mode=='main'else all_tf_recovery_root(sha);began=time.monotonic()
 def stop(*_):raise TimeoutError('PROVIDER_HLC_ALL_TF_LEARNING_SIGNAL_OR_WALL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,EXECUTION['internal_seconds'])
 try:
  guard=require_native_provider_hlc_all_tf_learning_execution(plan_sha256=sha,mode=mode);plan,binding,pref,live=(inspect_all_tf if mode=='main'else inspect_all_tf_resume)(sha,True);need(guard['plan']==binding and guard['output_root']==str(output),'ALL_TF_GUARD_PLAN_ROOT');output.mkdir(exist_ok=False);identity=core.future_cpu_process_identity()
  fields={'plan':binding,'policy_at_claim':pref,'runtime':live,'process_identity':identity,'remaining_invocations':0,'TRAIN_diagnostic_only':True,'backbone_gradient_policy':None}
  if mode=='resume':fields.update(original_plan=plan['metadata']['original_plan'],restored_checkpoint_receipt=plan['inputs']['checkpoint_receipt'],starting_cursor=plan['starting_cursor'])
  claim=pub(output,'CLAIM.json',fields)
 except BaseException:
  signal.setitimer(signal.ITIMER_REAL,0);raise
 try:
  allowed={plan['inputs'][key]['path']for key in ('feature_archive','target_arrays')};cursors=range(256,3694,256)if mode=='main'else plan['checkpoint_global_cursors']
  allowed|={str(output/name)for name in ['FINAL.pt','FINAL_LOGITS.npz']+['UPDATE%06d.pt'%j for j in cursors if j<3694]}
  if mode=='main':allowed.add(str(output/'INITIAL.pt'))
  else:allowed.add(plan['inputs']['checkpoint']['path'])
  if mode=='resume'and plan['schema_version']=='gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'and plan['starting_cursor']<plan['schedule']['logical_batches']:allowed.add(str(output/'TRAIN_END.pt'))
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('ALL_TF_LEARNING_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0])
    if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('ALL_TF_LEARNING_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit);report=(physical_all_tf if mode=='main'else physical_all_tf_resume)(plan,binding,pref,live);check_sources_inputs(plan)
  need(ref(binding['path'])==binding and ref(POLICY)==pref,'ALL_TF_SAME_FROZEN_PLAN_POLICY');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'ALL_TF_BOTH_BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'ALL_TF_M1_FIREWALL')
  result=pub(output,'RESULT.json',{'status':'COMPLETE_NONE_FULL_POPULATION_TRAIN_DIAGNOSTIC_EPOCH','mode':mode,'plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'source_unchanged':True,'elapsed_seconds':time.monotonic()-began,'test_data_used':False})
  terminal=pub(output,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'mode':mode,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  pub(output,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'mode':mode,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:1800],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)

def run_all_tf_resume(sha):
 return run_all_tf(sha,mode='resume')

def main_all_tf(mode='main'):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_learning_execution
 need(mode in ('main','resume'),'ALL_TF_CLI_MODE');flag='--all-tf-learning-plan-sha256'if mode=='main'else'--all-tf-learning-resume-plan-sha256';parser=argparse.ArgumentParser();parser.add_argument(flag,required=True,dest='sha');parser.add_argument('--preflight',action='store_true');args=parser.parse_args();need(sys.argv[1:]==[flag,args.sha]+(['--preflight']if args.preflight else[]),'ALL_TF_EXACT_CLI')
 if args.preflight:
  require_native_provider_hlc_all_tf_learning_execution(plan_sha256=args.sha,mode=mode);plan,binding,pref,live=(inspect_all_tf if mode=='main'else inspect_all_tf_resume)(args.sha,False)
  print(json.dumps({'status':'PREFLIGHT_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PASS','mode':mode,'plan':binding,'policy':pref,'payloads_decoded':0,'models':0,'CUDA_initialized':False,'output_created':False,'test_data_used':False}))
 else:print(json.dumps(run_all_tf(args.sha,mode=mode),sort_keys=True))

def main_all_tf_resume():
 return main_all_tf(mode='resume')
# End all-TF learning authority.

# BEGIN GX1 V3 BOUNDED DEVELOPMENT 20261127
DEV_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001'
DEV_PARENT='3ef40c291594a5bcc6d923ec5dcc4fd4e18ab746'
DEV_PROTOCOL=REPO/'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_PROTOCOL_20261127_001.json'
DEV_PHASES=('cohort','warm','cold','fit')
DEV_PARENT_STAGE='native_v3_bounded_development_test_20261127_001'
DEV_PLANS={phase:REPO/'configs/research'/(DEV_NAME+'_'+phase.upper()+('_002'if phase=='warm'else'_004'if phase=='fit'else'')+'.json')for phase in DEV_PHASES}
DEV_ROOTS={phase:Path('/home/andre2/GX1_RUNS')/(DEV_NAME+'_'+phase+('_004'if phase=='fit'else''))for phase in DEV_PHASES}
DEV_STAGES={phase:DEV_NAME.lower()+'_'+phase for phase in DEV_PHASES}
DEV_SCHEMA='gx1_native_all_tf_provider_hlc_v3_development_plan_v1'
DEV_TEST='tests/test_native_provider_hlc_v3_development_v1.py'
DEV_REVIEW='gx1/audit/native_all_tf_provider_hlc_review_v3.py'
DEV_GUARD_TEST='tests/test_native_all_tf_provider_hlc_v3_development_guard_v1.py'
DEV_ROW_SHA='84fca333ffd882d679e6339ea9663811b9d16564cb8f8617072753f4f6db5025'
DEV_ISSUE_SHA='ca9868f542d290045c2640c320d1e3517329c8a4223c8ec3a58a153d86a283f1'
DEV_FLAGS=(*FLAGS,'exit_training_and_evaluation_this_wave')

def development_reverse_source(raw):
 start=b'# BEGIN GX1 V3 BOUNDED DEVELOPMENT 20261127\n';end=b'# END GX1 V3 BOUNDED DEVELOPMENT 20261127\n'
 need(raw.count(start)==raw.count(end)==1,'DEV_SINGLE_ADDITIVE_BLOCK');first,tail=raw.split(start);_,last=tail.split(end)
 first=issue_mix_reverse_legacy(first)
 dispatch=b" if sys.argv[1:2]==['--v3-development-plan-sha256']:return main_v3_development()\n"
 if DEV_RECOVERY_DISPATCH in first:
  need(first.count(DEV_RECOVERY_DISPATCH)==1,'DEV_RECOVERY_EXACT_DISPATCH');first=first.replace(DEV_RECOVERY_DISPATCH,b'')
 if DIRECTION_GRADIENT_DISPATCH in first:
  need(first.count(DIRECTION_GRADIENT_DISPATCH)==1,'DIRECTION_GRADIENT_DISPATCH');first=first.replace(DIRECTION_GRADIENT_DISPATCH,b'')
 if BLOCK_ORDER_DISPATCH in first:
  need(first.count(BLOCK_ORDER_DISPATCH)==1,'BLOCK_ORDER_DISPATCH');first=first.replace(BLOCK_ORDER_DISPATCH,b'')
 if BLOCK_RECOVERY_DISPATCH in first:
  need(first.count(BLOCK_RECOVERY_DISPATCH)==1,'BLOCK_RECOVERY_DISPATCH');first=first.replace(BLOCK_RECOVERY_DISPATCH,b'')
 need(first.count(dispatch)==1,'DEV_EXACT_DISPATCH');return first.replace(dispatch,b'')+last

def development_source_bridge():
 before=subprocess.check_output(['git','show',DEV_PARENT+':'+SELF],cwd=REPO);after=(REPO/SELF).read_bytes()
 need(development_reverse_source(after)==before,'DEV_LEGACY_BYTES_UNCHANGED')
 return {'parent_commit':DEV_PARENT,'owner':SELF,'parent_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after).hexdigest(),'only_one_additive_block_and_dispatch':True,'baseline74_numeric_functions_changed':0}

def development_json_value(value):
 if type(value)in(type(None),bool,int,float,str):return value
 if type(value)in(list,tuple):return [development_json_value(x)for x in value]
 if type(value)is dict:
  need(all(type(k)is str for k in value),'DEV_JSON_STRING_KEYS');return {k:development_json_value(v)for k,v in value.items()}
 raise ValueError('PROVIDER_HLC_DEV_UNSUPPORTED_JSON:'+type(value).__name__)

def development_json_bytes(value):
 expected=development_json_value(value);raw=(json.dumps(expected,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 need(json.loads(raw)==expected,'DEV_JSON_PREPUBLICATION');return raw

def development_publish(root,name,value):
 raw=value if type(value)is bytes else development_json_bytes(value);path=root/name;_publish_bytes(path,raw)
 need(path.read_bytes()==raw,'DEV_PHYSICAL_READBACK');return ref(path)

def development_recipe(protocol):
 reference=read(protocol['references']['canonical_recipe_plan']);canonical=core.future_cpu_recipe(reference)
 paired=read(protocol['references']['canonical_paired_recipe'])['recipe']
 recipe={k:copy.deepcopy(canonical[k])for k in ('optimizer','gradient_clip_norm','scheduler','seed')}
 recipe.update({k:copy.deepcopy(paired[k])for k in ('train_batch_rows','eval_batch_rows','passes','checkpoint_every_optimizer_updates','allocator_phase_policy')})
 need(recipe==paired==protocol['recipe']and protocol['backbone_gradient_policy']is None and protocol['score_phases']==['INITIAL','PASS1','PASS2']and recipe['passes']==2 and recipe['train_batch_rows']==96 and recipe['eval_batch_rows']==256 and recipe['seed']==20260911 and recipe['scheduler']is None and recipe['gradient_clip_norm']==1. and recipe['checkpoint_every_optimizer_updates']==256,'DEV_EXACT_CANONICAL_PAIRED_RECIPE')
 need(protocol['fit_checkpoint_cursors']==[1,432]and protocol['technical_checkpoint_cursors']=={'warm':[1,2],'cold':[3]},'DEV_PAIRED_CHECKPOINT_CURSORS');return recipe

def development_selection(selection):
 need(selection['rows']==20727 and selection['nonempty_buckets']==1377 and selection['whole_TRAIN_rows']==354550 and selection['whole_TRAIN_issue_ns_sha256']==train.TRAIN_SHA and selection['selected_issue_ns_sha256']==DEV_ISSUE_SHA and selection['selected_source_rows_sha256']==DEV_ROW_SHA,'DEV_CANONICAL_SELECTION')
 need(selection['selector']=='bucket=floor((issue_ns-22h)/24h);min16_linspace_int64_including_endpoints'and selection['bucket_origin_ns']==train.ORIGIN and selection['bucket_duration_ns']==train.DAY and selection['target_filter']is False and selection['class_or_observability_quotas']is False and selection['top_up']is False,'DEV_CLOCK_ONLY_SELECTION')
 need(len(selection['bucket_sizes'])==len(selection['selected_per_bucket'])==1377 and sum(selection['bucket_sizes'])==354550 and sum(selection['selected_per_bucket'])==20727 and all(b==min(16,a)for a,b in zip(selection['bucket_sizes'],selection['selected_per_bucket'])),'DEV_ALL_BUCKETS_ENDPOINT_GEOMETRY');return selection

def development_gate(binding,phase,admitted=False):
 need(phase in DEV_PHASES,'DEV_PHASE')
 return {'status':'ADMITTED_EXACT_V3_DEVELOPMENT_'+phase.upper()if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted and phase!='cohort','test_authorized':False}

def development_validate_flags(policy,phase,admitted=False):
 wanted={k:False for k in DEV_FLAGS}
 if admitted and phase!='cohort':wanted.update(training_enabled=True,native_training_enabled=True)
 need(all(policy.get(k)is value for k,value in wanted.items()),'DEV_EXACT_PHASE_FLAGS')

def development_execution(phase,base):
 from gx1.contracts.gx1_capped_execution_v1 import _native_v3_development_execution
 return _native_v3_development_execution(phase)

def development_limits(phase):
 if phase=='cohort':return {'feature_cache_payload_loads':1,'target_archive_loads':1,'cohort_prior_fits':1,'factory_prior_fits':0,'normalization_fits':0,'model_constructions':0,'optimizer_constructions':0,'model_forwards':0,'optimizer_steps':0,'new_array_archives':0}
 from gx1.audit import native_provider_hlc_replay_v1 as replay
 if phase in('warm','cold'):
  value=copy.deepcopy(replay.ALL_LIMITS[phase]);value.update(checkpoint_prepublication_loads=2 if phase=='warm'else 1,logit_archive_writes=0,logit_archive_readbacks=0,normalization_fits=0,prior_fits=0,skipped_batches=0);return value
 return {'model_constructions':1,'optimizer_constructions':1,'feature_cache_payload_loads':1,'target_archive_loads':1,'model_forwards':678,'backbone_forwards':678,'readout_forwards':678,'optimizer_steps':432,'backward_calls':432,'autograd_calls':9,'skipped_batches':0,'diagnostic_skips':0,'checkpoint_writes':2,'checkpoint_readbacks':2,'checkpoint_receipt_writes':2,'checkpoint_prepublication_loads':2,'checkpoint_loads':0,'model_restores':0,'logit_archive_writes':3,'logit_archive_readbacks':3,'normalization_fits':0,'prior_fits':0}

def development_closed(phase):
 root=DEV_ROOTS[phase];terminal_ref=ref(root/'TERMINAL.json');terminal=read(terminal_ref);binding=terminal['plan'];result_ref=terminal['result'];result=read(result_ref)
 need(binding['path']==str(DEV_PLANS[phase])and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and result['status']=='COMPLETE_V3_DEVELOPMENT_'+phase.upper()and result['phase']==phase and result['plan']==binding and result['claim']==terminal['claim']and result['error']is None and all(result['bookends'].values()),'DEV_CLOSED_'+phase.upper())
 return {'plan':binding,'result':result_ref,'terminal':terminal_ref,'value':result}

def build_v3_development_plan(phase,runtime=None,source_commit=None):
 """Metadata/source only. Never load tensors, truth arrays, features or CUDA."""
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_v3_development_source_bridge
 need(phase in DEV_PHASES,'DEV_PHASE');protocol_ref=ref(DEV_PROTOCOL);protocol=read(protocol_ref);need(protocol['schema_version']=='gx1_native_all_tf_provider_hlc_v3_development_protocol_v1','DEV_PROTOCOL_SCHEMA');recipe=development_recipe(protocol)
 refs=protocol['references'];base=read(refs['V3_replay_plan']);initial=read(refs['V3_initial_acceptance']);qualified=read(refs['V3_replay_acceptance']);selection=development_selection(read(refs['canonical_clock_selection']))
 need(base['backbone_gradient_policy']is None and base['initial_reference']['initial_only']is True and base['initial_reference']['optimizer_updates']==0 and base['model_contract']['contract_sha256']=='ded7f2a05417d108ef8bed4d88f31d40ea1de21292e964586598ddaa818010c3'and base['initial_reference']['model_state_sha256']=='be5b6266739021319e5fdc9f87bde70aa987e56c0979b2bda1586302a48b5491'and base['initial_reference']['rng_sha256']=='343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d','DEV_OWN_FRESH_V3_FACTORY')
 need(initial['full_initialbaseline_qualified']is True and qualified['status']=='PASS_ALL_TF_PROVIDER_HLC_V3_OWN_C2_COLD_REPLAY'and qualified['all10_proofs_equal']is True and qualified['both_boots_source_inputs_policy_closed']is True,'DEV_QUALIFIED_ORIGINAL_FUNCTION')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head),'DEV_SOURCE_COMMIT');meta=copy.deepcopy(base['metadata'])
 if phase!='cohort':
  need(runtime is not None,'DEV_RUNTIME_REQUIRED');core.future_cuda_require_runtime(runtime);need(runtime['source_commit']==head and all(runtime[k]==meta['runtime'][k]for k in('gpu','library_versions')),'DEV_QUALIFIED_RUNTIME');meta['runtime']=copy.deepcopy(runtime)
 owners=dict(base['source_owners'])
 for name,digest in owners.items():
  if name not in GUARDS:need(ref(REPO/name)['sha256']==digest,'DEV_IMMUTABLE_BASELINE_OWNER:'+name)
 producer_hashes={name:base['source_owners'][name]for name in GUARDS};guard=native_provider_hlc_v3_development_source_bridge(producer_hashes,producer_commit=DEV_PARENT)
 owners.update({name:ref(REPO/name)['sha256']for name in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST,DEV_REVIEW)});inputs={**copy.deepcopy(base['inputs']),'development_protocol':protocol_ref,**{key:value for key,value in refs.items()}}
 parents={};schedule=None
 if phase!='cohort':
  parent=development_closed('cohort');report=parent['value']['report'];need(parent['value']['protocol']==protocol_ref,'DEV_SAME_COHORT_PROTOCOL');parents['cohort']={k:parent[k]for k in('plan','result','terminal')};inputs.update({'development_cohort_'+k:parent[k]for k in('plan','result','terminal')});inputs.update({k:report[k]for k in('selection_ref','cohort_prior','cohort_spec')});schedule=copy.deepcopy(report['schedule']if phase=='fit'else report['technical_schedule'])
 if phase in('cold','fit'):
  parent=development_closed('warm');need(parent['value']['protocol']==protocol_ref,'DEV_SAME_WARM_PROTOCOL');parents['warm']={k:parent[k]for k in('plan','result','terminal')};inputs.update({'development_warm_'+k:parent[k]for k in('plan','result','terminal')});inputs.update({k:parent['value']['report'][k]for k in('cursor2_receipt','cursor3_proof')})
 if phase=='fit':
  parent=development_closed('cold');report=parent['value']['report'];need(parent['value']['protocol']==protocol_ref and report['cold_replay_bit_exact']is True and len(report['full_cursor3_proof'])==10 and all(n>0 for n in report['restored_nonzero_moments'].values()),'DEV_OWN_SELECTED_COHORT_REPLAY');parents['cold']={k:parent[k]for k in('plan','result','terminal')};inputs.update({'development_cold_'+k:parent[k]for k in('plan','result','terminal')})
 if phase=='cold':inputs['selected_cursor2_checkpoint']=read(inputs['cursor2_receipt'])['checkpoint']
 metadata_roles=set(inputs)-{'feature_archive','target_arrays','selected_cursor2_checkpoint'}-{k for k in inputs if k.startswith('provider_raw_response_')};payload_roles=set(inputs)-metadata_roles
 return {'schema_version':DEV_SCHEMA,'phase':phase,'stage':DEV_STAGES[phase],'output_root':str(DEV_ROOTS[phase]),'source_commit':head,'source_owners':owners,'baseline_source_owners':base['source_owners'],'guard_source_bridge':guard,'owner_source_bridge':development_source_bridge(),'protocol':protocol_ref,'metadata':meta,'model_contract':base['model_contract'],'initial_reference':base['initial_reference'],'inputs':inputs,'metadata_input_roles':sorted(metadata_roles),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload_roles)},'parents':parents,'recipe':recipe,'schedule':schedule,'checkpoint_cursors':[]if phase=='cohort'else[1,432]if phase=='fit'else[1,2]if phase=='warm'else[3],'backbone_gradient_policy':None,'execution':development_execution(phase,base),'limits':development_limits(phase),'selection':selection,'scope':protocol['scope']}

def development_check_inputs(plan,physical=False):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit'],'DEV_FROZEN_CWD_BRANCH_HEAD')
 for name,digest in plan['source_owners'].items():need(ref(REPO/name)['sha256']==digest,'DEV_SOURCE_SHA:'+name)
 for role,binding in plan['inputs'].items():
  if physical or role in plan['metadata_input_roles']:need(ref(binding['path'])==binding,'DEV_INPUT_SHA:'+role)
  else:
   stat=Path(binding['path']).stat();need({'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}==plan['payload_stat'][role],'DEV_PAYLOAD_STAT:'+role)

def inspect_v3_development(sha,phase,admitted=False):
 binding={'path':str(DEV_PLANS[phase]),'sha256':sha};plan=read(binding)
 need(plan==build_v3_development_plan(phase,None if phase=='cohort'else plan['metadata']['runtime'],plan['source_commit']),'DEV_EXACT_PLAN');development_check_inputs(plan)
 pref=ref(POLICY);policy=read(pref);development_validate_flags(policy,phase,admitted);parent=policy.get(DEV_PARENT_STAGE,{});need(parent.get('protocol')==plan['protocol']and parent.get('phase_gates',{}).get(phase)==development_gate(binding,phase,admitted),'DEV_ONE_USE_PARENT_GATE');need(not DEV_ROOTS[phase].exists()and not DEV_ROOTS[phase].is_symlink(),'DEV_FRESH_ROOT')
 live=None
 if phase!='cohort':live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 return plan,binding,pref,live

def development_context(plan):
 """One existing V3 full loader, one unchanged clock selector, one exact slice."""
 context=adapter.load_context(plan['metadata']);clock=train.select_study_clock_rows({'provider':context['provider']});need(clock['manifest']==plan['selection'],'DEV_PHYSICAL_CLOCK_MANIFEST')
 take=np.searchsorted(context['positions'],clock['selected_positions']);need(np.array_equal(context['positions'][take],clock['selected_positions']),'DEV_CLOCK_POSITIONS')
 truth=train.slice_provider_hlc_truth(context['truth'],take,target_contract=context['target_contract'],fold_upper_ns=train.UPPER)
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 truth_sha=objective.native_future_state_truth_sha256(truth,target_contract=context['target_contract'])
 masks=objective._native_future_state_truth(truth,rows=20727,fold_upper_ns=train.UPPER,target_contract=context['target_contract']);batches=core.all_score_batches(np.arange(20727,dtype=np.int64),96);known=[int(masks['activity'][b].sum())for b in batches]
 schedule=train.make_schedule(clock['selected_source_rows'],truth['issue_ns'],batch_rows=96,passes=2,known_activity_by_batch=known)
 need(schedule['batches_per_pass']==216 and schedule['logical_batches']==schedule['optimizer_updates']==432 and schedule['skipped_batches']==0 and len(batches[-1])==87,'DEV_EXACT_SUPPORTED_SCHEDULE')
 technical=train.make_schedule(clock['selected_source_rows'][:288],truth['issue_ns'][:288],batch_rows=96,passes=1,known_activity_by_batch=known[:3]);need(technical['optimizer_updates']==3,'DEV_SUPPORTED_TECHNICAL_PREFIX')
 selected={**context,'positions':clock['selected_positions'],'source_rows':clock['selected_source_rows'],'truth':truth,'truth_sha256':truth_sha,'batches':batches,'schedule':schedule,'technical_schedule':technical}
 return selected

def development_cohort(plan,root):
 from gx1.audit import native_all_tf_provider_hlc_review_v3 as review
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 selection_ref=development_publish(root,'SELECTION.json',plan['selection'])
 context=development_context(plan);cohort_prior=objective.fit_native_future_state_train_prior(truth=context['truth'],fold_upper_ns=train.UPPER,target_contract=context['target_contract'])
 prior_ref=development_publish(root,'COHORT_SCORING_PRIOR.json',cohort_prior)
 spec=review.build_cohort_spec(whole_spec=read(plan['inputs']['whole_review_spec']),selection=plan['selection'],selection_ref=selection_ref,factory_prior=context['prior'],cohort_prior=cohort_prior);review.validate_cohort_spec(spec);spec_ref=development_publish(root,'COHORT_REVIEW_SPEC.json',spec)
 return {'selection_ref':selection_ref,'cohort_prior':prior_ref,'cohort_spec':spec_ref,'schedule':context['schedule'],'technical_schedule':context['technical_schedule'],'truth_sha256':context['truth_sha256'],'rows':20727,'nonempty_buckets':1377,'matched_scoring_prior_fits':1,'factory_prior_fits':0,'normalization_fits':0,'certificate_preserved':True,'counts':plan['limits']}

def development_nn_context(plan):
 from gx1.audit import native_all_tf_provider_hlc_review_v3 as review
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 context=development_context(plan);report=read(plan['inputs']['development_cohort_result'])['report'];selection_ref=plan['inputs']['selection_ref'];need(read(selection_ref)==plan['selection']and context['truth_sha256']==report['truth_sha256'],'DEV_QUALIFIED_SELECTED_TRUTH')
 prior=objective.require_native_future_state_train_prior(read(plan['inputs']['cohort_prior']),target_contract=context['target_contract']);need(prior['fit_population']['issue_rows']==20727 and prior['fit_population']['truth_sha256']==context['truth_sha256']and prior['fit_population']['issue_ns_sha256']==DEV_ISSUE_SHA,'DEV_SELECTED_PRIOR_POPULATION')
 need(plan['schedule']==(context['schedule']if plan['phase']=='fit'else context['technical_schedule']),'DEV_PHYSICAL_SCHEDULE')
 spec=read(plan['inputs']['cohort_spec']);review.validate_cohort_spec(spec);context.update(scoring_prior=prior,selection_ref=selection_ref,review_spec=spec);return context

def development_trajectory(plan):
 return train.seal({'schema_version':'gx1_v3_selected_development_trajectory_v1','protocol':plan['protocol'],'cohort_result':plan['inputs']['development_cohort_result'],'cohort_spec':plan['inputs']['cohort_spec'],'selection':plan['selection'],'model_contract':plan['model_contract'],'schedule':plan['schedule'],'recipe':plan['recipe'],'runtime_compatibility':{k:plan['metadata']['runtime'][k]for k in('gpu','library_versions')},'baseline_source_owners':plan['baseline_source_owners']})

def development_checkpoint(root,name,checkpoint,counts,execution_ref):
 import io,torch
 before=core.state_sha(core.training_rng());stream=io.BytesIO();torch.save(checkpoint,stream);raw=stream.getvalue();stream.close()
 loaded=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);counts['checkpoint_prepublication_loads']+=1
 need(core.state_sha(loaded)==core.state_sha(checkpoint)and train.checkpoint_proof(loaded)==train.checkpoint_proof(checkpoint),'DEV_CHECKPOINT_PREPUBLICATION');del loaded
 binding=development_publish(root,name,raw);counts['checkpoint_writes']+=1;loaded=torch.load(binding['path'],map_location='cpu',weights_only=True);counts['checkpoint_readbacks']+=1;need(core.state_sha(loaded)==core.state_sha(checkpoint),'DEV_CHECKPOINT_PHYSICAL_READBACK');del loaded
 receipt={'schema_version':'gx1_native_future_state_training_checkpoint_receipt_v1','checkpoint':binding,'original_plan':checkpoint['producer']['original_plan'],'execution_plan':execution_ref,'producer_lane':checkpoint['producer']['lane'],'producer_runtime':checkpoint['producer']['runtime'],'logical_cursor':checkpoint['position']['logical_cursor'],'optimizer_updates':checkpoint['position']['optimizer_updates'],'trajectory_sha256':checkpoint['trajectory']['contract_sha256'],'proof':train.checkpoint_proof(checkpoint)}
 result=development_publish(root,Path(name).with_suffix('.json').name,receipt);counts['checkpoint_receipt_writes']+=1;need(core.state_sha(core.training_rng())==before,'DEV_CHECKPOINT_RNG_UNCHANGED');return result

def development_logits(root,label,issues,outputs,counts):
 import io
 arrays={'issue_ns':issues,**outputs};memory=io.BytesIO();np.savez_compressed(memory,**arrays);raw=memory.getvalue();memory.close()
 def validate(saved):
  need(set(saved.files)==set(arrays),'DEV_LOGITS_INVENTORY')
  for name,value in arrays.items():
   other=saved[name];need(other.dtype==value.dtype and other.shape==value.shape and core.digest(other)==core.digest(value),'DEV_LOGITS_BYTES')
 with np.load(io.BytesIO(raw),allow_pickle=False)as saved:validate(saved)
 binding=development_publish(root,label+'_LOGITS.npz',raw);counts['logit_archive_writes']+=1
 with np.load(binding['path'],allow_pickle=False)as saved:validate(saved)
 counts['logit_archive_readbacks']+=1;return binding

def development_score(model,context,plan,root,label,counts):
 import torch
 from gx1.audit import native_provider_hlc_replay_v1 as replay
 from gx1.audit import native_all_tf_provider_hlc_review_v3 as review
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 captured={};records=[]
 def capture(module,args,output):
  for key,value in output.items():captured.setdefault(key,[]).append(value.detach().cpu().numpy().copy())
 hook=model.register_forward_hook(capture)
 try:
  for take in core.all_score_batches(np.arange(20727,dtype=np.int64),256):records.append(replay.score(model,adapter.packet(context,take),plan))
 finally:hook.remove()
 outputs={key:np.concatenate(value)for key,value in captured.items()};need(set(outputs)=={'activity_logits','side_logits'}and all(v.shape==(20727,5)and v.dtype==np.float32 and np.isfinite(v).all()for v in outputs.values()),'DEV_COMPLETE_SELECTED_OUTPUTS')
 cpu={key:torch.from_numpy(value)for key,value in outputs.items()};kwargs={'truth':context['truth'],'fold_upper_ns':train.UPPER,'target_contract':context['target_contract']};scores=objective.native_future_state_score_sums(model_output=cpu,**kwargs)
 constants={}
 for name,prior in [('factory',context['prior']),('cohort',context['scoring_prior'])]:
  logits=objective.native_future_state_train_prior_logits(prior,rows=20727,target_contract=context['target_contract']);constants[name]=objective.native_future_state_score_sums(model_output=logits,**kwargs)
  if label=='INITIAL'and name=='factory':need(all(torch.equal(cpu[key],value)for key,value in logits.items()),'DEV_INITIAL_EXACT_FACTORY_PRIOR')
 panel=review.review_cohort_outputs(truth=context['truth'],outputs=outputs,factory_prior=context['prior'],cohort_prior=context['scoring_prior'],selected_positions=context['positions'],source_rows=context['source_rows'],selection_ref=context['selection_ref'],spec=context['review_spec'],phase=label,input_binding=context['input_binding'],normalization=context['normalization'])
 logits_ref=development_logits(root,label,context['truth']['issue_ns'],outputs,counts);value={'phase':label,'rows':20727,'batches':81,'canonical_scores':scores,'constant_scores':constants,'logits':logits_ref,'review':panel,'truth_sha256':context['truth_sha256'],'certificate':context['truth']['target_observation_certificate_ref'],'batch_score_hashes':[record['output_sha256']for record in records]}
 return development_publish(root,label+'_SCORE.json',value)

def development_nn(plan,binding,root,live):
 import torch
 from gx1.audit import native_provider_hlc_replay_v1 as replay
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 phase=plan['phase'];context=development_nn_context(plan);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=plan['initial_reference'];need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'DEV_FRESH_FACTORY_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'DEV_FRESH_ADAMW');trajectory=development_trajectory(plan);steps=[];scores={};extra={}
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'DEV_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def checkpoint(cursor,name,original=binding,lane=phase):
  cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=original,lane=lane,cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);return development_checkpoint(root,name,cp,counts,binding)
 try:
  if phase=='warm':
   batches=context['batches'][:3];first=adapter.packet(context,batches[0]);extra['finite_check_gradient_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=False);extra['equal_TF_eval_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=True)
   for j,take in enumerate(batches):
    current=first if j==0 else adapter.packet(context,take);steps.append(train.training_step(model,optimizer,current,plan,counts,j));need(not steps[-1]['skip_update'],'DEV_PREFIX_SUPPORT')
    if j==0:extra['post_first_head_data_gradient']=core.future_cuda_gradient_probe(model,current,plan,counts);extra['cursor1_receipt']=checkpoint(1,'CURSOR001.pt')
    if j==1:extra['nonzero_moments_at_cursor2']=replay.nonzero_moments(model,optimizer);extra['cursor2_receipt']=checkpoint(2,'CURSOR002.pt');need(steps[-1]['backbone_data_gradient_l2']>0,'DEV_SELECTED_REAL_BACKBONE_GRADIENT')
   scores['CURSOR3']=replay.score(model,current,plan);cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=binding,lane='warm',cursor=3,steps=steps,scores=scores,producer_runtime=live);extra['cursor3_proof']=development_publish(root,'CURSOR003_PROOF.json',{'proof':train.checkpoint_proof(cp),'nonzero_moments':replay.nonzero_moments(model,optimizer)})
  elif phase=='cold':
   warm=read(plan['inputs']['development_warm_plan']);warm_ref=plan['inputs']['development_warm_plan'];warm_result=read(plan['inputs']['development_warm_result']);core.future_cpu_parent_stopped(warm_result['process_identity']);trajectory=development_trajectory(warm)
   need(trajectory==development_trajectory(plan),'DEV_SAME_SELECTED_TECHNICAL_TRAJECTORY');receipt=read(plan['inputs']['cursor2_receipt']);reference=read(plan['inputs']['cursor3_proof']);need(receipt['original_plan']==receipt['execution_plan']==warm_ref and receipt['producer_lane']=='warm'and receipt['logical_cursor']==receipt['optimizer_updates']==2 and receipt['trajectory_sha256']==trajectory['contract_sha256'],'DEV_AUTHENTIC_SELECTED_C2')
   need(receipt['checkpoint']==plan['inputs']['selected_cursor2_checkpoint']and ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'DEV_SELECTED_C2_BYTES_BEFORE_LOAD');cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
   train.strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=warm_ref,expected_cursor=2,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;extra['restored_nonzero_moments']=replay.nonzero_moments(model,optimizer);need(extra['restored_nonzero_moments']==warm_result['report']['nonzero_moments_at_cursor2'],'DEV_OWN_C2_MOMENT_COUNTS');steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);current=adapter.packet(context,context['batches'][2]);steps.append(train.training_step(model,optimizer,current,plan,counts,2));scores['CURSOR3']=replay.score(model,current,plan)
   next_cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=warm_ref,lane='cold',cursor=3,steps=steps,scores=scores,producer_runtime=live);proof=train.checkpoint_proof(next_cp);need(proof==reference['proof']and len(proof)==10,'DEV_SELECTED_ALL10_PROOFS_BITEXACT');extra.update(cold_replay_bit_exact=True,full_cursor3_proof=proof,replayed_checkpoint_receipt=development_checkpoint(root,'REPLAYED_CURSOR003.pt',next_cp,counts,binding))
  else:
   scores['INITIAL']=development_score(model,context,plan,root,'INITIAL',counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_INITIAL_SCORE_TO_TRAIN')
   for j in range(432):
    steps.append(train.training_step(model,optimizer,adapter.packet(context,context['batches'][j%216]),plan,counts,j));need(not steps[-1]['skip_update']and counts['optimizer_steps']==j+1,'DEV_CANONICAL_STEP')
    if j==0:extra['cursor1_receipt']=checkpoint(1,'CURSOR001.pt')
    if(j+1)%216==0:
     label='PASS'+str((j+1)//216);train.release_phase_allocator_cache(model,optimizer,label='DEV_TRAIN_TO_'+label);scores[label]=development_score(model,context,plan,root,label,counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_'+label+'_TO_NEXT_PHASE')
   before=core.state_sha(optimizer.state_dict());panels=train.diagnostic_panels(context['schedule']);need(panels['supported_panels']==3 and not panels['skipped_empty_batch_indices'],'DEV_FIXED_DIAGNOSTIC_PANELS');extra['gradient_diagnostics']=[]
   for index in panels['candidate_batch_indices']:extra['gradient_diagnostics'].append({'batch_index':index,'report':core.future_cuda_gradient_probe(model,adapter.packet(context,context['batches'][index]),plan,counts)})
   need(core.state_sha(optimizer.state_dict())==before,'DEV_DIAGNOSTICS_NO_OPTIMIZER_MUTATION');extra['final_checkpoint_receipt']=checkpoint(432,'FINAL.pt')
  need(counts==plan['limits'],'DEV_EXACT_PHYSICAL_COUNTS');need(torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'DEV_VRAM_BOUND')
  return {'counts':counts,'trajectory':trajectory,'fresh_initial_state_sha256':initial['model_state_sha256'],'fresh_initial_rng_sha256':initial['rng_sha256'],'model_contract_sha256':plan['model_contract']['contract_sha256'],'scores':scores,'optimizer_updates':counts['optimizer_steps'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN','technical_weights_never_fit_initializer':True,**extra}
 finally:
  for hook in hooks:hook.remove()

def development_bookends(plan,binding,pref,boot):
 checks={}
 tests={'plan':lambda:ref(DEV_PLANS[plan['phase']])==binding,'sources':lambda:all(ref(REPO/n)['sha256']==h for n,h in plan['source_owners'].items()),'inputs':lambda:all(ref(q['path'])==q for q in plan['inputs'].values()),'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def development_allowed_payloads(plan):
 phase=plan['phase'];root=DEV_ROOTS[phase];allowed={str(Path(plan['inputs'][role]['path']).resolve())for role in ('feature_archive','target_arrays')}
 names=[]if phase=='cohort'else['CURSOR001.pt','CURSOR002.pt']if phase=='warm'else['REPLAYED_CURSOR003.pt']if phase=='cold'else['CURSOR001.pt','FINAL.pt','INITIAL_LOGITS.npz','PASS1_LOGITS.npz','PASS2_LOGITS.npz']
 allowed.update(str(root/name)for name in names)
 if phase=='cold':allowed.add(str(Path(plan['inputs']['selected_cursor2_checkpoint']['path']).resolve()))
 return allowed

def development_io_audit(allowed,event,args):
 if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('V3_DEVELOPMENT_NETWORK_FORBIDDEN')
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  name=os.fsdecode(args[0])
  if name.endswith(('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'))and str(Path(name).resolve())not in allowed:raise RuntimeError('V3_DEVELOPMENT_PAYLOAD_PATH_FORBIDDEN')

def run_v3_development(sha,phase):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_all_tf_provider_hlc_v3_development_execution
 execution=require_native_all_tf_provider_hlc_v3_development_execution(plan_sha256=sha,phase=phase);plan,binding,pref,live=inspect_v3_development(sha,phase,True);root=DEV_ROOTS[phase];boot={'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()};started=time.monotonic();root.mkdir(exist_ok=False)
 claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'boots':boot,'phase':phase,'execution':execution,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('V3_DEVELOPMENT_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed=development_allowed_payloads(plan);sys.addaudithook(lambda event,args:development_io_audit(allowed,event,args))
  development_check_inputs(plan,physical=True);report=development_cohort(plan,root)if phase=='cohort'else development_nn(plan,binding,root,live)
  need(core.common.firewall_state()['denied_attempts']==0,'DEV_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=development_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'DEV_BOOKEND_FAILED')
  result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_DEVELOPMENT_'+phase.upper()if error is None else'FAILED_CONSUMED','phase':phase,'plan':binding,'protocol':plan['protocol'],'claim':claim,'process_identity':core.future_cpu_process_identity(),'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'})
  terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal,'phase':phase}

def main_v3_development():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_all_tf_provider_hlc_v3_development_execution
 parser=argparse.ArgumentParser();parser.add_argument('--v3-development-plan-sha256',required=True);parser.add_argument('--phase',choices=DEV_PHASES,required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_development_plan_sha256
 need(sys.argv[1:]==['--v3-development-plan-sha256',sha,'--phase',args.phase]+(['--preflight']if args.preflight else[]),'DEV_EXACT_CLI')
 if args.preflight:
  require_native_all_tf_provider_hlc_v3_development_execution(plan_sha256=sha,phase=args.phase);plan,binding,pref,live=inspect_v3_development(sha,args.phase,False);print(json.dumps({'status':'PREFLIGHT_V3_DEVELOPMENT_METADATA_ONLY','phase':args.phase,'plan':binding,'policy':pref,'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_development(sha,args.phase),sort_keys=True))

# BEGIN GX1 V3 EXACT RECOVERY 20261128
DEV_RECOVERY_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002'
DEV_RECOVERY_STAGE='native_v3_fit004_exact_recovery_20261128_002'
DEV_RECOVERY_LANES=('reference','cold','recovery')
DEV_RECOVERY_PLANS={x:REPO/'configs/research'/(DEV_RECOVERY_NAME+'_'+x.upper()+'.json')for x in DEV_RECOVERY_LANES}
DEV_RECOVERY_ROOTS={x:Path('/home/andre2/GX1_RUNS')/(DEV_RECOVERY_NAME+'_'+x)for x in DEV_RECOVERY_LANES}
DEV_RECOVERY_PROTOCOL=REPO/'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_PROTOCOL_20261128_002.json'
DEV_RECOVERY_ANCHOR='0330fe6c8bb0af4d1f107a3f86ed72ed98692518'
DEV_RECOVERY_ORIGINAL={'path':str(DEV_PLANS['fit']),'sha256':'ebf7222c9fb403cf0adc26738cc682c088035dc2b2dc2fc5aff603a741660437'}
DEV_RECOVERY_RECEIPT={'path':str(DEV_ROOTS['fit']/'CURSOR001.json'),'sha256':'adcc8717242aa5bac0505332fbe360dc3578315bb49f5e6230ebefe15e871529'}
DEV_RECOVERY_DISPATCH=b" if sys.argv[1:2]==['--v3-development-recovery-plan-sha256']:return main_v3_development_recovery()\n"
DEV_RECOVERY_REVERSE=b" if DEV_RECOVERY_DISPATCH in first:\n  need(first.count(DEV_RECOVERY_DISPATCH)==1,'DEV_RECOVERY_EXACT_DISPATCH');first=first.replace(DEV_RECOVERY_DISPATCH,b'')\n"

def recovery_source_bridge():
 raw=(REPO/SELF).read_bytes();a=b'\n# BEGIN GX1 V3 EXACT RECOVERY 20261128\n';b=b'# END GX1 V3 EXACT RECOVERY 20261128\n'
 need(raw.count(a)==raw.count(b)==1,'RECOVERY_ONE_BLOCK');first,tail=raw.split(a);_,last=tail.split(b);raw=first+last
 raw=issue_mix_reverse_legacy(raw)
 if BLOCK_RECOVERY_DISPATCH in raw:
  need(raw.count(BLOCK_RECOVERY_DISPATCH)==raw.count(BLOCK_RECOVERY_LEGACY_REVERSE)==1,'BLOCK_RECOVERY_OUTER_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'')
 if BLOCK_ORDER_DISPATCH in raw:
  need(raw.count(BLOCK_ORDER_DISPATCH)==raw.count(BLOCK_ORDER_LEGACY_REVERSE)==1,'BLOCK_ORDER_OUTER_REVERSAL');raw=raw.replace(BLOCK_ORDER_DISPATCH,b'').replace(BLOCK_ORDER_LEGACY_REVERSE,b'')
 if DIRECTION_GRADIENT_DISPATCH in raw:
  need(raw.count(DIRECTION_GRADIENT_DISPATCH)==raw.count(DIRECTION_GRADIENT_REVERSE)==1,'DIRECTION_GRADIENT_RECOVERY_REVERSAL');raw=raw.replace(DIRECTION_GRADIENT_DISPATCH,b'').replace(DIRECTION_GRADIENT_REVERSE,b'')
 need(raw.count(DEV_RECOVERY_DISPATCH)==raw.count(DEV_RECOVERY_REVERSE)==1,'RECOVERY_EXACT_DISPATCH_REVERSE')
 raw=raw.replace(DEV_RECOVERY_DISPATCH,b'').replace(DEV_RECOVERY_REVERSE,b'');old=subprocess.check_output(['git','show',DEV_RECOVERY_ANCHOR+':'+SELF],cwd=REPO)
 need(raw==old,'RECOVERY_ORIGINAL_LEAF_BYTES_UNCHANGED')
 return {'producer_commit':DEV_RECOVERY_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'original_development_nn_and_numeric_paths_unchanged':True,'only_exact_recovery_block_dispatch_reversal':True}

def recovery_gate(binding,lane,admitted=False):
 need(lane in DEV_RECOVERY_LANES,'RECOVERY_LANE')
 return {'status':'ADMITTED_EXACT_V3_RECOVERY_'+lane.upper()if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False}

def recovery_require_gate(value,binding,lane,admitted=False):
 expected=recovery_gate(binding,lane,admitted);booleans=('execution_authorized','relaunch_allowed','training_authorized','test_authorized')
 need(type(value)is dict and set(value)==set(expected)and type(value['remaining_invocations'])is int and all(type(value[k])is bool for k in booleans),'RECOVERY_GATE_TYPES')
 need(value==expected,'RECOVERY_EXACT_GATE');return value

def recovery_admission_snapshot(lane,binding,protocol,policy_ref):
 snapshot_ref=ref(E/('V3_FIT004_EXACT_RECOVERY_'+lane.upper()+'_ADMITTED_POLICY_20261128_002.json'));snapshot=read(snapshot_ref)
 need(policy_ref=={'path':str(POLICY),'sha256':snapshot_ref['sha256']},'RECOVERY_ADMISSION_SNAPSHOT_SHA');development_validate_flags(snapshot,'fit',True);parent=snapshot.get(DEV_RECOVERY_STAGE,{})
 recovery_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,True)
 need(parent.get('protocol')==protocol and parent.get('original_plan')==DEV_RECOVERY_ORIGINAL and parent.get('checkpoint_receipt')==DEV_RECOVERY_RECEIPT and parent.get('execution_authorized')is True and parent.get('active_lane')==lane and all(parent.get(k)is False for k in('main_fit_authorized','test_authorized','spending_authorized')),'RECOVERY_ADMITTED_SCOPE')
 return snapshot_ref

def recovery_limits(lane,original):
 need(lane in DEV_RECOVERY_LANES,'RECOVERY_LANE');schedule=original['schedule'];cursor=1;total=schedule['optimizer_updates'];n=schedule['batches_per_pass'];passes=original['recipe']['passes'];rows=original['selection']['rows'];e=(rows+original['recipe']['eval_batch_rows']-1)//original['recipe']['eval_batch_rows'];panels=train.diagnostic_panels(schedule)
 need(total==schedule['logical_batches']==n*passes==432 and n==216 and rows==20727 and e==81 and panels['supported_panels']==3,'RECOVERY_DERIVED_ORIGINAL_SCOPE')
 limits=dict.fromkeys(original['limits'],0);updates=2 if lane=='reference'else 1 if lane=='cold'else total-cursor;f=updates+(passes*e+panels['supported_panels']if lane=='recovery'else 0)
 limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,target_archive_loads=1,model_forwards=f,backbone_forwards=f,readout_forwards=f,optimizer_steps=updates,backward_calls=updates,checkpoint_writes=1,checkpoint_readbacks=1,checkpoint_receipt_writes=1,checkpoint_prepublication_loads=1)
 if lane!='reference':limits.update(checkpoint_loads=1,model_restores=1)
 if lane=='recovery':limits.update(autograd_calls=original['limits']['autograd_calls'],logit_archive_writes=passes,logit_archive_readbacks=passes)
 return limits

def recovery_original_metadata():
 original=read(DEV_RECOVERY_ORIGINAL);receipt=read(DEV_RECOVERY_RECEIPT);root=DEV_ROOTS['fit'];trajectory=development_trajectory(original)
 need(original['phase']=='fit'and original['output_root']==str(root)and original['backbone_gradient_policy']is None and original['recipe']==development_recipe(read(original['protocol']))and original['limits']==development_limits('fit'),'RECOVERY_ORIGINAL_FIXED_FIT')
 refs={k:ref(root/n)for k,n in [('original_claim','CLAIM.json'),('original_result','RESULT.json'),('original_terminal','TERMINAL.json'),('original_INITIAL_score','INITIAL_SCORE.json'),('original_INITIAL_logits','INITIAL_LOGITS.npz')]}
 refs.update(original_plan=DEV_RECOVERY_ORIGINAL,checkpoint_receipt=DEV_RECOVERY_RECEIPT,original_checkpoint=receipt['checkpoint'],original_admitted_policy=ref(E/'V3_DEVELOPMENT_FIT004_ADMITTED_POLICY_20261128_001.json'),original_stop_acceptance=ref(E/'V3_DEVELOPMENT_FIT004_STOP_ACCEPTANCE_20261128_001.json'),recovery_design=ref(E/'V3_DEVELOPMENT_FIT004_EXACT_RECOVERY_DESIGN_20261128_001.json'))
 claim=read(refs['original_claim']);result=read(refs['original_result']);terminal=read(refs['original_terminal']);admission=read(refs['original_admitted_policy']);stop=read(refs['original_stop_acceptance'])
 need(claim['plan']==result['plan']==terminal['plan']==DEV_RECOVERY_ORIGINAL and result['claim']==terminal['claim']==refs['original_claim']and terminal['result']==refs['original_result']and result['status']==terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False,'RECOVERY_CONSUMED_ORIGINAL_CHAIN')
 need(set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values())and result['phase']==claim['phase']=='fit'and result['protocol']==original['protocol'],'RECOVERY_ORIGINAL_BOOKENDS')
 need(claim['policy_at_claim']=={'path':str(POLICY),'sha256':refs['original_admitted_policy']['sha256']}and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False,'RECOVERY_ORIGINAL_CLAIM_POLICY')
 development_validate_flags(admission,'fit',True);need(admission[DEV_PARENT_STAGE]['phase_gates']['fit']==development_gate(DEV_RECOVERY_ORIGINAL,'fit',True),'RECOVERY_ORIGINAL_ADMISSION')
 need(stop['result']==refs['original_result']and stop['terminal']==refs['original_terminal']and stop['all6_bookends_PASS']is True and stop['durable_cursor_receipt']==DEV_RECOVERY_RECEIPT and stop['durable_checkpoint']==receipt['checkpoint']and stop['initial_score']==refs['original_INITIAL_score']and stop['initial_all_scores_and_logit_SHA_equal_FIT002_factory_reference']is True,'RECOVERY_STOP_BINDINGS')
 keys={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime','logical_cursor','optimizer_updates','trajectory_sha256','proof'}
 need(set(receipt)==keys and receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'and receipt['original_plan']==receipt['execution_plan']==DEV_RECOVERY_ORIGINAL and receipt['producer_lane']=='fit'and type(receipt['logical_cursor'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==1 and receipt['checkpoint']['path']==str(root/'CURSOR001.pt'),'RECOVERY_AUTHENTIC_C1_RECEIPT')
 proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(receipt['trajectory_sha256']==trajectory['contract_sha256']and proof['trajectory']==core.state_sha(trajectory)and proof['position']==core.state_sha(train.position(original['schedule'],1))and proof['scores']==core.state_sha({'INITIAL':refs['original_INITIAL_score']})and proof['scheduler_state']==core.state_sha(None)and proof['training']==core.state_sha(True),'RECOVERY_C1_SEMANTIC_IDENTITY')
 core.future_cuda_runtime_matches(original['metadata']['runtime'],receipt['producer_runtime']);core.future_cpu_parent_stopped(claim['process_identity'])
 need(claim['boots']=={k:receipt['producer_runtime'][k]for k in('physical_windows_boot_id','wsl_boot_id')},'RECOVERY_ORIGINAL_PRODUCER_BOOTS')
 for n,digest in original['baseline_source_owners'].items():
  if n not in GUARDS:need(ref(REPO/n)['sha256']==digest,'RECOVERY_BASELINE_SOURCE:'+n)
 meta=original['metadata'];binding=read(meta['input_binding_ref']);normalization=read(meta['normalization'])
 need(binding==meta['input_binding']and binding['input_normalization_sha256']==normalization['contract_sha256']and binding['feature_provider_contract_sha256']==normalization['feature_provider_contract_sha256'],'RECOVERY_PHYSICAL_FEATURE_NORMALIZATION')
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 contract=native_future_state_model_contract(binding,architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy=None)
 need(contract==original['model_contract'],'RECOVERY_PHYSICAL_CHECKPOINT_MODEL_CONTRACT')
 initial=read(refs['original_INITIAL_score']);need(initial['phase']=='INITIAL'and initial['logits']==refs['original_INITIAL_logits'],'RECOVERY_ORIGINAL_INITIAL_LOGITS')
 identity={'input_binding_ref':meta['input_binding_ref'],'normalization_ref':meta['normalization'],'model_contract_sha256':contract['contract_sha256'],'target_contract_sha256':meta['target_contract']['contract_sha256'],'actual_metadata_contract_recomputed':True,'payloads_decoded':0}
 return original,receipt,trajectory,refs,identity

def build_v3_development_recovery_protocol():
 original,receipt,trajectory,refs,identity=recovery_original_metadata()
 return {'schema_version':'gx1_native_v3_exact_recovery_protocol_v1','experiment_id':DEV_RECOVERY_NAME,'original_plan':DEV_RECOVERY_ORIGINAL,'checkpoint_receipt':DEV_RECOVERY_RECEIPT,'original_protocol':original['protocol'],'original_trajectory':trajectory,'original_lineage':refs,'physical_input_identity':identity,'recipe':original['recipe'],'schedule':original['schedule'],'limits':{lane:recovery_limits(lane,original)for lane in DEV_RECOVERY_LANES},'scope':{'same_original_two_pass_trajectory':True,'technical_weights_never_recovery_initializer':True,'uncheckpointed_original_updates':'UNKNOWN','TEST_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}}

def recovery_closed(lane):
 root=DEV_RECOVERY_ROOTS[lane];binding=ref(DEV_RECOVERY_PLANS[lane]);tr=ref(root/'TERMINAL.json');terminal=read(tr);rr=terminal['result'];result=read(rr)
 need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and terminal['plan']==result['plan']==binding and terminal['claim']==result['claim']and result['status']=='COMPLETE_V3_RECOVERY_'+lane.upper()and result['lane']==lane and result['error']is None and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'RECOVERY_CLOSED_'+lane.upper())
 claim=read(result['claim']);need(ref(result['claim']['path'])==result['claim']and claim['plan']==binding and claim['lane']==lane and claim['original_plan']==DEV_RECOVERY_ORIGINAL and claim['checkpoint_receipt']==DEV_RECOVERY_RECEIPT and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False,'RECOVERY_PARENT_CLAIM')
 need(result['report']['original_plan']==DEV_RECOVERY_ORIGINAL and result['report']['original_checkpoint_receipt']==DEV_RECOVERY_RECEIPT,'RECOVERY_PARENT_LINEAGE');core.future_cpu_parent_stopped(result['process_identity'])
 snapshot=recovery_admission_snapshot(lane,binding,read(binding)['protocol'],claim['policy_at_claim']);need(claim['admitted_policy_snapshot']==snapshot,'RECOVERY_PARENT_ADMISSION_SNAPSHOT')
 return {'plan':binding,'result':rr,'terminal':tr,'claim':result['claim'],'admitted_policy_snapshot':snapshot,'value':result}

def build_v3_development_recovery_plan(lane,runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_v3_development_recovery_source_bridge
 need(lane in DEV_RECOVERY_LANES,'RECOVERY_LANE');pr=ref(DEV_RECOVERY_PROTOCOL);protocol=read(pr);need(protocol==build_v3_development_recovery_protocol(),'RECOVERY_EXACT_COMMON_PROTOCOL')
 original=read(DEV_RECOVERY_ORIGINAL);receipt=read(DEV_RECOVERY_RECEIPT);trajectory=protocol['original_trajectory'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head),'RECOVERY_SOURCE_COMMIT')
 core.future_cuda_require_runtime(runtime);need(runtime['source_commit']==head,'RECOVERY_RUNTIME_SOURCE');train.runtime_compatible(trajectory['runtime_compatibility'],runtime)
 inputs={**copy.deepcopy(original['inputs']),**protocol['original_lineage'],'recovery_protocol':pr};parents={}
 for parent_lane in (()if lane=='reference'else('reference',)if lane=='cold'else('reference','cold')):
  closed=recovery_closed(parent_lane);parent=read(closed['plan']);need(parent['metadata']['trajectory']==trajectory and parent['protocol']==pr,'RECOVERY_FROZEN_PARENT_FUNCTION')
  parents[parent_lane]={k:closed[k]for k in('plan','result','terminal','claim','admitted_policy_snapshot')};inputs.update({parent_lane+'_'+k:closed[k]for k in('plan','result','terminal','claim','admitted_policy_snapshot')});report=closed['value']['report']
  if parent_lane=='reference':
   need(report['original_C1_all10_equal']is True,'RECOVERY_REFERENCE_ORIGINAL_C1');inputs['reference_cursor2_receipt']=report['cursor2_receipt'];reference=read(report['cursor2_receipt'])
   need(reference['original_plan']==reference['execution_plan']==closed['plan']and reference['producer_lane']=='reference'and reference['logical_cursor']==reference['optimizer_updates']==2 and reference['trajectory_sha256']==trajectory['contract_sha256'],'RECOVERY_HONEST_REFERENCE_PRODUCER')
  else:need(report['cold_replay_bit_exact']is True and report['all10_proofs_equal']is True,'RECOVERY_PHYSICAL_COLD_PROOF')
 owners=dict(original['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,digest in owners.items():need(ref(REPO/n)['sha256']==digest,'RECOVERY_SOURCE:'+n)
 for parent_lane in parents:
  parent=read(parents[parent_lane]['plan']);need(parent['source_owners']==owners and parent['source_commit']==head,'RECOVERY_PARENT_SOURCES_FROZEN')
 old={n:hashlib.sha256(subprocess.check_output(['git','show',DEV_RECOVERY_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS};metadata_roles=set(inputs)-{'feature_archive','target_arrays','original_checkpoint','original_INITIAL_logits'}-{k for k in inputs if k.startswith('provider_raw_response_')};payload=set(inputs)-metadata_roles
 return {'schema_version':'gx1_native_all_tf_provider_hlc_v3_development_recovery_plan_v1','lane':lane,'stage':(DEV_RECOVERY_NAME+'_'+lane).lower(),'source_commit':head,'output_root':str(DEV_RECOVERY_ROOTS[lane]),'source_owners':owners,'guard_source_bridge':native_provider_hlc_v3_development_recovery_source_bridge(old,producer_commit=DEV_RECOVERY_ANCHOR),'owner_source_bridge':recovery_source_bridge(),'protocol':pr,'original_plan':DEV_RECOVERY_ORIGINAL,'checkpoint_receipt':DEV_RECOVERY_RECEIPT,'metadata':{'original':original,'receipt':receipt,'trajectory':trajectory,'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(metadata_roles),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'parents':parents,'recipe':original['recipe'],'schedule':original['schedule'],'model_contract':original['model_contract'],'initial_reference':original['initial_reference'],'backbone_gradient_policy':None,'starting_cursor':0 if lane=='reference'else 1,'ending_cursor':original['schedule']['logical_batches']if lane=='recovery'else 2,'limits':protocol['limits'][lane],'execution':development_execution('fit'if lane=='recovery'else'warm'if lane=='reference'else'cold',original),'physical_input_identity':protocol['physical_input_identity'],'scope':protocol['scope']}

def recovery_check_inputs(plan,physical=False):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit'],'RECOVERY_FROZEN_CWD_BRANCH_HEAD')
 for n,digest in plan['source_owners'].items():need(ref(REPO/n)['sha256']==digest,'RECOVERY_SOURCE_SHA:'+n)
 for role,binding in plan['inputs'].items():
  if physical or role in plan['metadata_input_roles']:need(ref(binding['path'])==binding,'RECOVERY_INPUT_SHA:'+role)
  else:
   s=Path(binding['path']).stat();need({'size':s.st_size,'mtime_ns':s.st_mtime_ns}==plan['payload_stat'][role],'RECOVERY_PAYLOAD_STAT:'+role)

def inspect_v3_development_recovery(sha,lane,admitted=False):
 binding={'path':str(DEV_RECOVERY_PLANS[lane]),'sha256':sha};plan=read(binding);need(plan==build_v3_development_recovery_plan(lane,plan['metadata']['runtime'],plan['source_commit']),'RECOVERY_EXACT_PLAN');recovery_check_inputs(plan)
 pref=ref(POLICY);policy=read(pref);development_validate_flags(policy,'fit',admitted);parent=policy.get(DEV_RECOVERY_STAGE,{});recovery_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,admitted)
 need(parent.get('protocol')==plan['protocol']and parent.get('original_plan')==DEV_RECOVERY_ORIGINAL and parent.get('checkpoint_receipt')==DEV_RECOVERY_RECEIPT and parent.get('lane_gates',{}).get(lane)==recovery_gate(binding,lane,admitted),'RECOVERY_ONE_USE_PARENT')
 if admitted:recovery_admission_snapshot(lane,binding,plan['protocol'],pref)
 need(not DEV_RECOVERY_ROOTS[lane].exists()and not DEV_RECOVERY_ROOTS[lane].is_symlink(),'RECOVERY_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 return plan,binding,pref,live

def recovery_restore(model,optimizer,checkpoint,plan,live):
 receipt=plan['metadata']['receipt'];original=plan['metadata']['original'];trajectory=plan['metadata']['trajectory']
 need(checkpoint['producer']=={'original_plan':DEV_RECOVERY_ORIGINAL,'lane':'fit','runtime':receipt['producer_runtime']},'RECOVERY_ACTUAL_ORIGINAL_PRODUCER')
 need(checkpoint['scores']=={'INITIAL':plan['inputs']['original_INITIAL_score']},'RECOVERY_ORIGINAL_INITIAL_REFERENCE')
 restored=train.strict_restore(model,optimizer,checkpoint,trajectory=trajectory,original_plan_ref=DEV_RECOVERY_ORIGINAL,expected_cursor=1,current_runtime=live,expected_state_proof=receipt['proof'])
 need(restored==train.position(original['schedule'],1),'RECOVERY_EXACT_RESTORE_POSITION');return restored

def recovery_allowed_payloads(plan):
 lane=plan['lane'];root=DEV_RECOVERY_ROOTS[lane];allowed={str(Path(plan['inputs'][r]['path']).resolve())for r in('feature_archive','target_arrays','original_checkpoint','original_INITIAL_logits')};names=['REFERENCE_CURSOR002.pt']if lane=='reference'else['REPLAYED_CURSOR002.pt']if lane=='cold'else['FINAL.pt','PASS1_LOGITS.npz','PASS2_LOGITS.npz']
 allowed.update(str(root/n)for n in names)
 if lane!='reference':allowed.add(str(Path(plan['inputs']['original_checkpoint']['path']).resolve()))
 return allowed

def recovery_nn(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 lane=plan['lane'];original=plan['metadata']['original'];trajectory=plan['metadata']['trajectory'];context=development_nn_context(original);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;initial=original['initial_reference']
 need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'RECOVERY_FRESH_FACTORY_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'RECOVERY_FRESH_ADAMW');steps=[];scores={'INITIAL':plan['inputs']['original_INITIAL_score']};extra={}
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'RECOVERY_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def checkpoint(cursor,name):
  cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=binding if lane=='reference'else DEV_RECOVERY_ORIGINAL,lane=lane,cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live)
  return cp,development_checkpoint(root,name,cp,counts,binding)
 try:
  if lane=='reference':
   for j in range(2):
    steps.append(train.training_step(model,optimizer,adapter.packet(context,context['batches'][j]),original,counts,j));need(not steps[-1]['skip_update']and counts['optimizer_steps']==j+1,'RECOVERY_REFERENCE_STEPS')
    if j==0:
     cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=binding,lane='reference',cursor=1,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);proof=train.checkpoint_proof(cp);need(proof==plan['metadata']['receipt']['proof'],'RECOVERY_INDEPENDENT_C1_ALL10')
     extra['cursor1_proof']=development_publish(root,'REFERENCE_CURSOR001_PROOF.json',{'proof':proof,'actual_producer':cp['producer'],'execution_plan':binding,'original_trajectory_plan':DEV_RECOVERY_ORIGINAL,'original_C1_receipt':DEV_RECOVERY_RECEIPT,'semantic_all10_equal':True});extra['original_C1_all10_equal']=True;del cp
   cp,extra['cursor2_receipt']=checkpoint(2,'REFERENCE_CURSOR002.pt');extra['full_cursor2_proof']=train.checkpoint_proof(cp)
  else:
   need(ref(plan['inputs']['original_checkpoint']['path'])==plan['inputs']['original_checkpoint'],'RECOVERY_C1_SHA_BEFORE_LOAD');cp=torch.load(plan['inputs']['original_checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
   restored=recovery_restore(model,optimizer,cp,plan,live);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
   extra['restore_proof']=development_publish(root,'RESTORE001_PROOF.json',{'canonical_strict_restore_completed':True,'original_plan':DEV_RECOVERY_ORIGINAL,'execution_plan':binding,'checkpoint_receipt':DEV_RECOVERY_RECEIPT,'checkpoint':plan['inputs']['original_checkpoint'],'position':restored,'expected_state_proof':plan['metadata']['receipt']['proof'],'new_model_forwards':0,'actual_original_producer_enforced':True})
   if lane=='cold':
    steps.append(train.training_step(model,optimizer,adapter.packet(context,context['batches'][1]),original,counts,1));need(not steps[-1]['skip_update']and counts['optimizer_steps']==1,'RECOVERY_COLD_STEP')
    cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=DEV_RECOVERY_ORIGINAL,lane='cold',cursor=2,steps=steps,scores=scores,producer_runtime=live);proof=train.checkpoint_proof(cp);reference=read(plan['inputs']['reference_cursor2_receipt']);need(proof==reference['proof']and len(proof)==10,'RECOVERY_COLD_C2_ALL10')
    extra.update(cold_replay_bit_exact=True,all10_proofs_equal=True,full_cursor2_proof=proof,cursor2_receipt=development_checkpoint(root,'REPLAYED_CURSOR002.pt',cp,counts,binding))
   else:
    total=original['schedule']['logical_batches'];n=original['schedule']['batches_per_pass']
    for j in range(1,total):
     steps.append(train.training_step(model,optimizer,adapter.packet(context,context['batches'][j%n]),original,counts,j));need(not steps[-1]['skip_update']and counts['optimizer_steps']+1==j+1 and len(steps)==j+1,'RECOVERY_ORIGINAL_CURSOR')
     if(j+1)%n==0:
      label='PASS'+str((j+1)//n);train.release_phase_allocator_cache(model,optimizer,label='DEV_TRAIN_TO_'+label);scores[label]=development_score(model,context,original,root,label,counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_'+label+'_TO_NEXT_PHASE')
    before=core.state_sha(optimizer.state_dict());panels=train.diagnostic_panels(context['schedule']);need(panels['supported_panels']==3 and not panels['skipped_empty_batch_indices'],'RECOVERY_ORIGINAL_DIAGNOSTICS');extra['gradient_diagnostics']=[]
    for index in panels['candidate_batch_indices']:extra['gradient_diagnostics'].append({'batch_index':index,'report':core.future_cuda_gradient_probe(model,adapter.packet(context,context['batches'][index]),original,counts)})
    need(core.state_sha(optimizer.state_dict())==before,'RECOVERY_DIAGNOSTICS_NO_OPTIMIZER_MUTATION');_,extra['final_checkpoint_receipt']=checkpoint(total,'FINAL.pt')
  need(counts==plan['limits'],'RECOVERY_EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'RECOVERY_VRAM_BOUND')
  return {'counts':counts,'trajectory':trajectory,'original_plan':DEV_RECOVERY_ORIGINAL,'execution_plan':binding,'original_checkpoint_receipt':DEV_RECOVERY_RECEIPT,'starting_cursor':plan['starting_cursor'],'ending_cursor':plan['ending_cursor'],'new_optimizer_updates':counts['optimizer_steps'],'logical_original_optimizer_updates':plan['ending_cursor'],'original_uncheckpointed_updates':'UNKNOWN','scores':scores,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN','technical_weights_never_recovery_initializer':True,**extra}
 finally:
  for hook in hooks:hook.remove()

def recovery_bookends(plan,binding,pref,boot):
 tests={'plan':lambda:ref(DEV_RECOVERY_PLANS[plan['lane']])==binding,'sources':lambda:all(ref(REPO/n)['sha256']==h for n,h in plan['source_owners'].items()),'inputs':lambda:all(ref(q['path'])==q for q in plan['inputs'].values())and recovery_admission_snapshot(plan['lane'],binding,plan['protocol'],pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};checks={}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def run_v3_development_recovery(sha,lane):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_all_tf_provider_hlc_v3_development_recovery_execution
 execution=require_native_all_tf_provider_hlc_v3_development_recovery_execution(plan_sha256=sha,lane=lane);plan,binding,pref,live=inspect_v3_development_recovery(sha,lane,True);root=DEV_RECOVERY_ROOTS[lane];boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();root.mkdir(exist_ok=False)
 claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'boots':boot,'lane':lane,'original_plan':DEV_RECOVERY_ORIGINAL,'checkpoint_receipt':DEV_RECOVERY_RECEIPT,'admitted_policy_snapshot':recovery_admission_snapshot(lane,binding,plan['protocol'],pref),'execution':execution,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('V3_RECOVERY_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed=recovery_allowed_payloads(plan);sys.addaudithook(lambda event,args:development_io_audit(allowed,event,args));recovery_check_inputs(plan,physical=True);report=recovery_nn(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'RECOVERY_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=recovery_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'RECOVERY_BOOKEND_FAILED')
  result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_RECOVERY_'+lane.upper()if error is None else'FAILED_CONSUMED','lane':lane,'plan':binding,'protocol':plan['protocol'],'original_plan':DEV_RECOVERY_ORIGINAL,'claim':claim,'process_identity':core.future_cpu_process_identity(),'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'})
  terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal,'lane':lane}

def main_v3_development_recovery():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_all_tf_provider_hlc_v3_development_recovery_execution
 parser=argparse.ArgumentParser();parser.add_argument('--v3-development-recovery-plan-sha256',required=True);parser.add_argument('--lane',choices=DEV_RECOVERY_LANES,required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_development_recovery_plan_sha256
 need(sys.argv[1:]==['--v3-development-recovery-plan-sha256',sha,'--lane',args.lane]+(['--preflight']if args.preflight else[]),'RECOVERY_EXACT_CLI')
 if args.preflight:
  require_native_all_tf_provider_hlc_v3_development_recovery_execution(plan_sha256=sha,lane=args.lane);plan,binding,pref,live=inspect_v3_development_recovery(sha,args.lane,False);print(json.dumps({'status':'PREFLIGHT_V3_EXACT_RECOVERY_METADATA_ONLY','lane':args.lane,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_development_recovery(sha,args.lane),sort_keys=True))

# BEGIN GX1 V3 DIRECTION GRADIENT 20261128
DIRECTION_GRADIENT_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001'
DIRECTION_GRADIENT_STAGE=DIRECTION_GRADIENT_NAME.lower()
DIRECTION_GRADIENT_PLAN=REPO/'configs/research'/(DIRECTION_GRADIENT_NAME+'.json')
DIRECTION_GRADIENT_ROOT=Path('/home/andre2/GX1_RUNS')/DIRECTION_GRADIENT_NAME
DIRECTION_GRADIENT_ANCHOR='70b975652f96ccb47792a35235827aca7e1b849e'
DIRECTION_GRADIENT_DESIGN={'path':str(E/'V3_DIRECTION_GRADIENT_DESIGN_20261128_001.json'),'sha256':'86c1b9675b29ad2f058fb9424afbd07f79ed796ab45a78fb360b9eb670f56e75'}
DIRECTION_GRADIENT_POLICY_SNAPSHOT=E/'V3_DIRECTION_GRADIENT_ADMITTED_POLICY_20261128_001.json'
DIRECTION_GRADIENT_DISPATCH=b" if sys.argv[1:2]==['--v3-direction-gradient-plan-sha256']:return main_v3_direction_gradient()\n"
DIRECTION_GRADIENT_REVERSE=b" if DIRECTION_GRADIENT_DISPATCH in first:\n  need(first.count(DIRECTION_GRADIENT_DISPATCH)==1,'DIRECTION_GRADIENT_DISPATCH');first=first.replace(DIRECTION_GRADIENT_DISPATCH,b'')\n"
DIRECTION_GRADIENT_RECOVERY_REVERSE=b" if DIRECTION_GRADIENT_DISPATCH in raw:\n  need(raw.count(DIRECTION_GRADIENT_DISPATCH)==raw.count(DIRECTION_GRADIENT_REVERSE)==1,'DIRECTION_GRADIENT_RECOVERY_REVERSAL');raw=raw.replace(DIRECTION_GRADIENT_DISPATCH,b'').replace(DIRECTION_GRADIENT_REVERSE,b'')\n"

def direction_gradient_source_bridge():
 raw=(REPO/SELF).read_bytes();a=b'\n# BEGIN GX1 V3 DIRECTION GRADIENT 20261128\n';b=b'# END GX1 V3 DIRECTION GRADIENT 20261128\n';need(raw.count(a)==raw.count(b)==1,'DIRECTION_GRADIENT_ONE_BLOCK');first,tail=raw.split(a);_,last=tail.split(b);raw=first+last
 raw=issue_mix_reverse_legacy(raw)
 if BLOCK_RECOVERY_DISPATCH in raw:
  need(raw.count(BLOCK_RECOVERY_DISPATCH)==raw.count(BLOCK_RECOVERY_LEGACY_REVERSE)==raw.count(BLOCK_RECOVERY_RAW_REVERSE)==1,'BLOCK_RECOVERY_DIRECTION_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'').replace(BLOCK_RECOVERY_RAW_REVERSE,b'')
 if BLOCK_ORDER_DISPATCH in raw:
  need(raw.count(BLOCK_ORDER_DISPATCH)==raw.count(BLOCK_ORDER_LEGACY_REVERSE)==raw.count(BLOCK_ORDER_RAW_REVERSE)==1,'BLOCK_ORDER_DIRECTION_REVERSAL');raw=raw.replace(BLOCK_ORDER_DISPATCH,b'').replace(BLOCK_ORDER_LEGACY_REVERSE,b'').replace(BLOCK_ORDER_RAW_REVERSE,b'')
 for fragment in (DIRECTION_GRADIENT_DISPATCH,DIRECTION_GRADIENT_REVERSE,DIRECTION_GRADIENT_RECOVERY_REVERSE):
  need(raw.count(fragment)==1,'DIRECTION_GRADIENT_EXACT_NORMALIZATION');raw=raw.replace(fragment,b'')
 old=subprocess.check_output(['git','show',DIRECTION_GRADIENT_ANCHOR+':'+SELF],cwd=REPO);need(raw==old,'DIRECTION_GRADIENT_OLD_LEAF_UNCHANGED')
 return {'producer_commit':DIRECTION_GRADIENT_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'only_additive_diagnostic_dispatch_and_exact_bridge_normalization':True,'old_numeric_and_recovery_functions_unchanged':True}

def direction_gradient_gate(binding,admitted=False):
 return {'status':'ADMITTED_EXACT_V3_DIRECTION_GRADIENT'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}

def direction_gradient_require_gate(value,binding,admitted=False):
 expected=direction_gradient_gate(binding,admitted)
 need(type(value)is dict and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized'))and type(value.get('remaining_invocations'))is int and all(value.get(k)==v for k,v in expected.items()),'DIRECTION_GRADIENT_EXACT_GATE')
 return value

def direction_gradient_admission_snapshot(binding,policy_ref):
 sr=ref(DIRECTION_GRADIENT_POLICY_SNAPSHOT);p=read(sr);need(policy_ref=={'path':str(POLICY),'sha256':sr['sha256']}and all(p.get(k)is False for k in DEV_FLAGS),'DIRECTION_GRADIENT_SNAPSHOT_FLAGS_SHA')
 direction_gradient_require_gate(p.get(DIRECTION_GRADIENT_STAGE,{}),binding,True)
 need(p[DIRECTION_GRADIENT_STAGE].get('design')==DIRECTION_GRADIENT_DESIGN,'DIRECTION_GRADIENT_SNAPSHOT_DESIGN');return sr

def direction_gradient_final_metadata():
 original,c1,trajectory,lineage,identity=recovery_original_metadata();design=read(DIRECTION_GRADIENT_DESIGN);closed=recovery_closed('recovery');producer=read(closed['plan']);report=closed['value']['report'];fr=report['final_checkpoint_receipt'];receipt=read(fr)
 need(design['final_checkpoint_receipt']==fr and design['recovery_result']==closed['result']and design['fixed_panels']==[0,107,214]and design['maximum_model_forwards']==3 and design['maximum_autograd_calls']==30 and design['optimizer_updates']==design['checkpoint_writes']==design['normalization_fits']==design['prior_fits']==0,'DIRECTION_GRADIENT_BOUND_DESIGN')
 need(fr['path']==str(DEV_RECOVERY_ROOTS['recovery']/'FINAL.json')and receipt['original_plan']==DEV_RECOVERY_ORIGINAL and receipt['execution_plan']==closed['plan']and receipt['producer_lane']=='recovery'and type(receipt['logical_cursor'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==original['schedule']['logical_batches']==432 and receipt['checkpoint']['path']==str(DEV_RECOVERY_ROOTS['recovery']/'FINAL.pt'),'DIRECTION_GRADIENT_AUTHENTIC_FINAL')
 core.future_cuda_runtime_matches(producer['metadata']['runtime'],receipt['producer_runtime'])
 need(producer['metadata']['original']==original and producer['metadata']['trajectory']==trajectory and report['ending_cursor']==report['logical_original_optimizer_updates']==432 and report['new_optimizer_updates']==431 and report['scores']['INITIAL']==lineage['original_INITIAL_score'],'DIRECTION_GRADIENT_FINAL_PRODUCER')
 proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof['trajectory']==core.state_sha(trajectory)and proof['position']==core.state_sha(train.position(original['schedule'],432))and proof['scores']==core.state_sha(report['scores'])and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'DIRECTION_GRADIENT_FINAL_SEMANTICS')
 panels=train.diagnostic_panels(original['schedule']);need(panels['candidate_batch_indices']==design['fixed_panels']and panels['supported_panels']==3 and not panels['skipped_empty_batch_indices'],'DIRECTION_GRADIENT_FIXED_PANELS')
 refs={**lineage,'design':DIRECTION_GRADIENT_DESIGN,'final_execution_plan':closed['plan'],'final_result':closed['result'],'final_terminal':closed['terminal'],'final_claim':closed['claim'],'final_admitted_policy':closed['admitted_policy_snapshot'],'final_checkpoint_receipt':fr,'final_checkpoint':receipt['checkpoint'],'final_acceptance':design['recovery_acceptance'],'final_backup':design['final_backup']}
 refs.update({'original_'+label+'_score':binding for label,binding in report['scores'].items()})
 for binding in refs.values():
  if Path(binding['path']).suffix not in('.pt','.npz','.npy'):read(binding)
 return original,receipt,trajectory,producer,report['scores'],refs,identity

def build_v3_direction_gradient_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_v3_direction_gradient_source_bridge
 original,receipt,trajectory,producer,scores,refs,identity=direction_gradient_final_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'DIRECTION_GRADIENT_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(trajectory['runtime_compatibility'],runtime)
 owners=dict(producer['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,d in owners.items():need(ref(REPO/n)['sha256']==d,'DIRECTION_GRADIENT_SOURCE:'+n)
 inputs={**copy.deepcopy(original['inputs']),**refs};metadata_roles={k for k,v in inputs.items()if Path(v['path']).suffix not in('.pt','.npz','.npy')and not k.startswith('provider_raw_response_')};payload=set(inputs)-metadata_roles
 limits=dict.fromkeys(development_limits('fit'),0);limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,target_archive_loads=1,checkpoint_loads=1,model_restores=1,model_forwards=3,backbone_forwards=3,readout_forwards=3,autograd_calls=30)
 old={n:hashlib.sha256(subprocess.check_output(['git','show',DIRECTION_GRADIENT_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 return {'schema_version':'gx1_native_v3_direction_gradient_plan_v1','stage':DIRECTION_GRADIENT_STAGE,'source_commit':head,'output_root':str(DIRECTION_GRADIENT_ROOT),'source_owners':owners,'guard_source_bridge':native_v3_direction_gradient_source_bridge(old,producer_commit=DIRECTION_GRADIENT_ANCHOR),'owner_source_bridge':direction_gradient_source_bridge(),'design':DIRECTION_GRADIENT_DESIGN,'original_plan':DEV_RECOVERY_ORIGINAL,'checkpoint_receipt':refs['final_checkpoint_receipt'],'metadata':{'original':original,'receipt':receipt,'trajectory':trajectory,'producer':producer,'scores':scores,'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(metadata_roles),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'physical_input_identity':identity,'model_contract':original['model_contract'],'recipe':original['recipe'],'panel_indices':[0,107,214],'limits':limits,'execution':development_execution('warm',original),'scope':read(DIRECTION_GRADIENT_DESIGN)['scientific_limits']}

def inspect_v3_direction_gradient(sha,admitted=False):
 binding={'path':str(DIRECTION_GRADIENT_PLAN),'sha256':sha};plan=read(binding);need(plan==build_v3_direction_gradient_plan(plan['metadata']['runtime'],plan['source_commit']),'DIRECTION_GRADIENT_EXACT_PLAN');recovery_check_inputs(plan)
 pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in DEV_FLAGS),'DIRECTION_GRADIENT_ALL6_FLAGS_CLOSED');parent=policy.get(DIRECTION_GRADIENT_STAGE,{})
 direction_gradient_require_gate(parent,binding,admitted);need(parent.get('design')==DIRECTION_GRADIENT_DESIGN,'DIRECTION_GRADIENT_PARENT_DESIGN')
 if admitted:direction_gradient_admission_snapshot(binding,pref)
 need(not DIRECTION_GRADIENT_ROOT.exists()and not DIRECTION_GRADIENT_ROOT.is_symlink(),'DIRECTION_GRADIENT_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 return plan,binding,pref,live

def direction_gradient_parameter_groups(named):
 from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
 groups={name:[]for name in(*MODEL_NATIVE_TRAINING_SPECIALISTS,'unassigned_shared_mix')};groups['all_backbone']=[];readout={}
 expected={f'readout.{task}_heads.{tf}.{part}'for task in('activity','side')for tf in train.TF for part in('bias','weight')}
 for index,(name,p)in enumerate(named):
  if name.startswith('backbone.'):
   matches=[f for f in MODEL_NATIVE_TRAINING_SPECIALISTS if f in name];groups[matches[0]if len(matches)==1 else'unassigned_shared_mix'].append(index);groups['all_backbone'].append(index)
  else:need(name in expected,'DIRECTION_GRADIENT_PARAMETER_OWNER');readout[name]=index
 need(set(readout)==expected and groups['all_backbone'],'DIRECTION_GRADIENT_EXACT_READOUT_GROUPS')
 return groups,readout

def direction_gradient_pair_measures(left,right,left_task,right_task):
 raw=core.shared_gradient_measures(left,right)
 return {'left_task':left_task,'right_task':right_task,'scope':'pair_only_left_plus_right_not_all_ten_tasks','left_l2':raw['g5_l2'],'right_l2':raw['gaux_l2'],'cross_dot':raw['cross_dot'],'cosine':raw['cosine'],'left_dot_pair_sum':raw['g5_dot_joint'],'raw_pair_sum_opposes_left':raw['raw_joint_opposes_M5'],'unused_left_parameter_tensors':raw['unused_M5_parameter_tensors'],'unused_right_parameter_tensors':raw['unused_aux_parameter_tensors']}

def direction_gradient_probe(model,optimizer,packet,original,counts):
 import torch
 from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
 named=tuple((n,p)for n,p in model.named_parameters()if p.requires_grad);parameters=tuple(p for _,p in named);groups,readout=direction_gradient_parameter_groups(named)
 need(all(p.grad is None for p in model.parameters()),'DIRECTION_GRADIENT_INITIAL_GRAD_NONE')
 before={'model':core.state_sha(model.state_dict()),'optimizer':core.state_sha(optimizer.state_dict()),'rng':core.state_sha(core.training_rng())};rng=core.training_rng();mode=model.training;model.eval()
 try:
  output,_,_=core.future_cuda_forward(model,packet['features']);loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=train.UPPER,target_contract=original['metadata']['target_contract']);n=loss['count'];need(n>0,'DIRECTION_GRADIENT_ACTIVITY_SUPPORT')
  tasks=[(task,tf,j)for task in('activity','side')for j,tf in enumerate(train.TF)];derivatives=[];task_reports={}
  for i,(task,tf,j)in enumerate(tasks):
   term=loss['loss_sums'][task][j]/n;g=torch.autograd.grad(term,parameters,retain_graph=i<len(tasks)-1,allow_unused=True);cpu=tuple(None if x is None else x.detach().to('cpu',copy=True)for x in g);del g;counts['autograd_calls']+=1;support=int(loss['counts'][task][j]);need(support>0 or all(x is None or not bool(torch.count_nonzero(x))for x in cpu),'DIRECTION_GRADIENT_UNKNOWN_TASK_ZERO')
   derivatives.append(cpu);norms={key:core.shared_gradient_measures(tuple(cpu[k]for k in indices),(None,)*len(indices))['g5_l2']for key,indices in groups.items()};head={}
   for name,index in readout.items():
    value=cpu[index];head[name]={'gradient_l2':0. if value is None else float(value.double().square().sum().sqrt()),'unused':value is None}
    if name.endswith('.bias')and value is not None:need(value.numel()==1,'DIRECTION_GRADIENT_SCALAR_BIAS');head[name]['signed_gradient']=float(value.item())
   task_reports[task+'.'+tf]={'support':support,'status':'SUPPORTED'if support else'UNKNOWN_NO_OBSERVED_TASK_SUPPORT','loss_sum':float(loss['loss_sums'][task][j].detach()),'original_denominator':n,'shared_group_l2':norms,'readout':head}
  pairwise={}
  for group,indices in groups.items():
   cells={}
   for i,(ta,tfa,_)in enumerate(tasks):
    left=ta+'.'+tfa
    for j in range(i+1,len(tasks)):
     tb,tfb,_=tasks[j];right=tb+'.'+tfb
     cells[left+'__'+right]={'status':'UNKNOWN_TASK_SUPPORT'}if not task_reports[left]['support']or not task_reports[right]['support']else{'status':'SUPPORTED','raw_data_gradients':direction_gradient_pair_measures(tuple(derivatives[i][k]for k in indices),tuple(derivatives[j][k]for k in indices),left,right)}
   pairwise[group]=cells
  observed={'model':core.state_sha(model.state_dict())==before['model'],'optimizer':core.state_sha(optimizer.state_dict())==before['optimizer'],'rng_before_cleanup':core.state_sha(core.training_rng())==before['rng'],'grad_none':all(p.grad is None for p in model.parameters())};need(all(observed.values()),'DIRECTION_GRADIENT_OBSERVED_STATE_MUTATION')
  return {'denominator_known_activity_cells':n,'support':{k:v.detach().cpu().tolist()for k,v in loss['counts'].items()},'per_task':task_reports,'shared_group_pairwise':pairwise,'parameter_names_sha256':core.state_sha([name for name,_ in named]),'group_parameter_tensor_counts':{k:len(v)for k,v in groups.items()},'observed_before_cleanup':observed,'temporary_eval_mode':True,'original_training_mode':mode,'scope':'Local unclipped data gradients; not AdamW updates, featureutility or optimaltaskweights. D1/macro parameters belong session_regime_encoder.'}
 finally:
  model.train(mode);core.restore_training_rng(rng)
  need(core.state_sha(model.state_dict())==before['model']and core.state_sha(optimizer.state_dict())==before['optimizer']and core.state_sha(core.training_rng())==before['rng']and model.training is mode and all(p.grad is None for p in model.parameters()),'DIRECTION_GRADIENT_CLEANUP_STATE')

def direction_gradient_physical(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 original=plan['metadata']['original'];receipt=plan['metadata']['receipt'];trajectory=plan['metadata']['trajectory'];counts=dict.fromkeys(plan['limits'],0);context=development_nn_context(original);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'DIRECTION_GRADIENT_FRESH_FACTORY')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'DIRECTION_GRADIENT_FRESH_ADAMW');need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'DIRECTION_GRADIENT_FINAL_SHA_BEFORE_LOAD');cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 need(cp['producer']=={'original_plan':DEV_RECOVERY_ORIGINAL,'lane':'recovery','runtime':receipt['producer_runtime']}and cp['scores']==plan['metadata']['scores'],'DIRECTION_GRADIENT_ACTUAL_FINAL_PRODUCER_SCORES')
 restored=train.strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=DEV_RECOVERY_ORIGINAL,expected_cursor=432,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp;need(restored==train.position(original['schedule'],432),'DIRECTION_GRADIENT_FINAL_CURSOR')
 def proof():
  return train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=DEV_RECOVERY_ORIGINAL,lane='diagnostic',cursor=432,steps=steps,scores=scores,producer_runtime=live))
 need(proof()==receipt['proof']and all(p.grad is None for p in model.parameters()),'DIRECTION_GRADIENT_ALL10_BEFORE_PROBES')
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'DIRECTION_GRADIENT_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))];reports=[]
 try:
  for index in plan['panel_indices']:
   packet=adapter.packet(context,context['batches'][index]);report=direction_gradient_probe(model,optimizer,packet,original,counts);need(proof()==receipt['proof'],'DIRECTION_GRADIENT_ALL10_AFTER_PANEL');reports.append({'batch_index':index,'report':report})
  need(proof()==receipt['proof']and counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'DIRECTION_GRADIENT_FINAL_COUNTS_STATE_VRAM')
  return {'counts':counts,'original_plan':DEV_RECOVERY_ORIGINAL,'final_checkpoint_receipt':plan['checkpoint_receipt'],'final_state_proof':receipt['proof'],'all10_semantic_proofs_unchanged':True,'panels':reports,'optimizer_updates':0,'new_checkpoint_writes':0,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN','scientific_limits':plan['scope']}
 finally:
  for hook in hooks:hook.remove()

def direction_gradient_bookends(plan,binding,pref,boot):
 checks={};tests={'plan':lambda:ref(DIRECTION_GRADIENT_PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==d for n,d in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and direction_gradient_admission_snapshot(binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def run_v3_direction_gradient(sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_direction_gradient_execution
 execution=require_native_v3_direction_gradient_execution(plan_sha256=sha);plan,binding,pref,live=inspect_v3_direction_gradient(sha,True);root=DIRECTION_GRADIENT_ROOT;root.mkdir(exist_ok=False);boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':direction_gradient_admission_snapshot(binding,pref),'execution':execution,'boots':boot,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('DIRECTION_GRADIENT_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed={str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};sys.addaudithook(lambda e,a:development_io_audit(allowed,e,a));recovery_check_inputs(plan,physical=True);report=direction_gradient_physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'DIRECTION_GRADIENT_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=direction_gradient_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'DIRECTION_GRADIENT_BOOKEND_FAILED')
  result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_DIRECTION_GRADIENT'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal}

def main_v3_direction_gradient():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_direction_gradient_execution
 parser=argparse.ArgumentParser();parser.add_argument('--v3-direction-gradient-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();need(sys.argv[1:]==['--v3-direction-gradient-plan-sha256',a.v3_direction_gradient_plan_sha256]+(['--preflight']if a.preflight else[]),'DIRECTION_GRADIENT_EXACT_CLI')
 if a.preflight:
  require_native_v3_direction_gradient_execution(plan_sha256=a.v3_direction_gradient_plan_sha256);plan,binding,pref,live=inspect_v3_direction_gradient(a.v3_direction_gradient_plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_V3_DIRECTION_GRADIENT_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'final_checkpoint_receipt':plan['checkpoint_receipt'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_direction_gradient(a.v3_direction_gradient_plan_sha256),sort_keys=True))

# BEGIN GX1 V3 DIRECTION BLOCK ORDER 20261128
BLOCK_ORDER_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001'
BLOCK_ORDER_STAGE=BLOCK_ORDER_NAME.lower()
BLOCK_ORDER_PLAN=REPO/'configs/research'/(BLOCK_ORDER_NAME+'.json')
BLOCK_ORDER_ROOT=Path('/home/andre2/GX1_RUNS')/BLOCK_ORDER_NAME
BLOCK_ORDER_ANCHOR='6297d53e28e7362a9561c8cc6ef26c9122f1822c'
BLOCK_ORDER_DESIGN={'path':str(E/'V3_DIRECTION_BLOCK_ORDER_DESIGN_20261128_001.json'),'sha256':'573e747a4ebea2f91a228a5853f4f16eb5a7411280dc62832d46a3e43f250942'}
BLOCK_ORDER_POLICY_SNAPSHOT=E/'V3_DIRECTION_BLOCK_ORDER_ADMITTED_POLICY_20261128_001.json'
BLOCK_ORDER_DISPATCH=b" if sys.argv[1:2]==['--v3-direction-block-order-plan-sha256']:return main_v3_direction_block_order()\n"
BLOCK_ORDER_LEGACY_REVERSE=b" if BLOCK_ORDER_DISPATCH in first:\n  need(first.count(BLOCK_ORDER_DISPATCH)==1,'BLOCK_ORDER_DISPATCH');first=first.replace(BLOCK_ORDER_DISPATCH,b'')\n"
BLOCK_ORDER_RAW_REVERSE=b" if BLOCK_ORDER_DISPATCH in raw:\n  need(raw.count(BLOCK_ORDER_DISPATCH)==raw.count(BLOCK_ORDER_LEGACY_REVERSE)==1,'BLOCK_ORDER_OUTER_REVERSAL');raw=raw.replace(BLOCK_ORDER_DISPATCH,b'').replace(BLOCK_ORDER_LEGACY_REVERSE,b'')\n"

BLOCK_ORDER_DIRECTION_REVERSE=b" if BLOCK_ORDER_DISPATCH in raw:\n  need(raw.count(BLOCK_ORDER_DISPATCH)==raw.count(BLOCK_ORDER_LEGACY_REVERSE)==raw.count(BLOCK_ORDER_RAW_REVERSE)==1,'BLOCK_ORDER_DIRECTION_REVERSAL');raw=raw.replace(BLOCK_ORDER_DISPATCH,b'').replace(BLOCK_ORDER_LEGACY_REVERSE,b'').replace(BLOCK_ORDER_RAW_REVERSE,b'')\n"

def block_order_source_bridge():
 raw=(REPO/SELF).read_bytes();a=b'\n# BEGIN GX1 V3 DIRECTION BLOCK ORDER 20261128\n';b=b'# END GX1 V3 DIRECTION BLOCK ORDER 20261128\n';need(raw.count(a)==raw.count(b)==1,'BLOCK_ORDER_ONE_BLOCK');first,tail=raw.split(a);_,last=tail.split(b);raw=first+last
 raw=issue_mix_reverse_legacy(raw)
 if BLOCK_RECOVERY_DISPATCH in raw:
  need(all(raw.count(f)==1 for f in (BLOCK_RECOVERY_DISPATCH,BLOCK_RECOVERY_LEGACY_REVERSE,BLOCK_RECOVERY_RAW_REVERSE,BLOCK_RECOVERY_DIRECTION_REVERSE)),'BLOCK_RECOVERY_ORDER_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'').replace(BLOCK_RECOVERY_RAW_REVERSE,b'').replace(BLOCK_RECOVERY_DIRECTION_REVERSE,b'')
 need(all(raw.count(fragment)==1 for fragment in (BLOCK_ORDER_DISPATCH,BLOCK_ORDER_LEGACY_REVERSE,BLOCK_ORDER_RAW_REVERSE,BLOCK_ORDER_DIRECTION_REVERSE)),'BLOCK_ORDER_EXACT_NORMALIZATIONS')
 raw=raw.replace(BLOCK_ORDER_DISPATCH,b'').replace(BLOCK_ORDER_LEGACY_REVERSE,b'').replace(BLOCK_ORDER_RAW_REVERSE,b'').replace(BLOCK_ORDER_DIRECTION_REVERSE,b'');old=subprocess.check_output(['git','show',BLOCK_ORDER_ANCHOR+':'+SELF],cwd=REPO);need(raw==old,'BLOCK_ORDER_OLD_LEAF_UNCHANGED')
 return {'producer_commit':BLOCK_ORDER_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'only_order_route_dispatch_and_exact_legacy_bridge_normalization':True,'all_old_numeric_functions_unchanged':True}

def block_order_gate(binding,admitted=False):
 return {'status':'ADMITTED_EXACT_V3_DIRECTION_BLOCK_ORDER'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False}

def block_order_require_gate(value,binding,admitted=False):
 expected=block_order_gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized'))and all(value.get(k)==v for k,v in expected.items()),'BLOCK_ORDER_EXACT_GATE');return value

def block_order_flags(policy,admitted=False):
 need(all(policy.get(k)is(admitted and k in DEV_FLAGS[:2])for k in DEV_FLAGS),'BLOCK_ORDER_EXACT_FLAGS')

def block_order_admission_snapshot(binding,pref):
 sr=ref(BLOCK_ORDER_POLICY_SNAPSHOT);p=read(sr);need(pref=={'path':str(POLICY),'sha256':sr['sha256']},'BLOCK_ORDER_SNAPSHOT_SHA');block_order_flags(p,True);parent=p.get(BLOCK_ORDER_STAGE,{})
 block_order_require_gate(parent,binding,True);need(parent.get('design')==BLOCK_ORDER_DESIGN,'BLOCK_ORDER_SNAPSHOT_DESIGN');return sr

def block_order_schedule(original):
 canonical=original['schedule'];n=canonical['batches_per_pass'];passes=original['recipe']['passes'];batch_rows=original['recipe']['train_batch_rows'];rows=canonical['rows']
 need(n==216 and passes==2 and rows==20727 and batch_rows==96 and rows-(n-1)*batch_rows==87 and canonical['logical_batches']==canonical['optimizer_updates']==n*passes and canonical['skipped_batches']==0 and all(v is True for v in canonical['update_flags']),'BLOCK_ORDER_ORIGINAL_SCHEDULE')
 private=np.random.Generator(np.random.PCG64(original['recipe']['seed']));permutations=[private.permutation(n).tolist()for _ in range(passes)];ids=[v for order in permutations for v in order]
 result={k:copy.deepcopy(v)for k,v in canonical.items()if k!='contract_sha256'};result.update(schema_version='gx1_native_v3_whole_batch_order_schedule_v1',canonical_schedule=canonical,generator='private_numpy_Generator_PCG64',seed=original['recipe']['seed'],shuffle=True,permutations=permutations,logical_original_batch_ids=ids,ordered_batch_source_rows_sha256=[canonical['batch_source_rows_sha256'][i]for i in ids],ordered_batch_issue_ns_sha256=[canonical['batch_issue_ns_sha256'][i]for i in ids],original_batch_rows=[min(batch_rows,rows-i*batch_rows)for i in range(n)])
 return train.seal(result)

def block_order_validate_schedule(schedule,original):
 need(type(schedule)is dict and type(schedule.get('permutations'))is list and len(schedule['permutations'])==original['recipe']['passes'],'BLOCK_ORDER_SCHEDULE_SCHEMA')
 n=original['schedule']['batches_per_pass']
 for order in schedule['permutations']:need(type(order)is list and len(order)==n and all(type(i)is int for i in order)and sorted(order)==list(range(n)),'BLOCK_ORDER_BIJECTION')
 need(schedule==block_order_schedule(original),'BLOCK_ORDER_EXACT_PRIVATE_ORDER');return schedule

def block_order_step_binding(schedule,index):
 need(type(index)is int and 0<=index<schedule['logical_batches'],'BLOCK_ORDER_LOGICAL_INDEX');batch=schedule['logical_original_batch_ids'][index]
 return {'schedule_sha256':schedule['contract_sha256'],'logical_index':index,'pass_index':index//schedule['batches_per_pass'],'offset_in_pass':index%schedule['batches_per_pass'],'original_batch_id':batch,'rows':schedule['original_batch_rows'][batch],'source_rows_sha256':schedule['ordered_batch_source_rows_sha256'][index],'issue_ns_sha256':schedule['ordered_batch_issue_ns_sha256'][index]}

def block_order_cursor(schedule,cursor):
 position=train.position(schedule,cursor)
 return {'position':position,'schedule_sha256':schedule['contract_sha256'],'next_original_batch':None if cursor==schedule['logical_batches']else block_order_step_binding(schedule,cursor)}

def block_order_trajectory(original,schedule):
 value={k:copy.deepcopy(v)for k,v in development_trajectory(original).items()if k!='contract_sha256'};value.update(schema_version='gx1_v3_order_only_training_trajectory_v1',schedule=schedule,original_trajectory_sha256=development_trajectory(original)['contract_sha256'],order_design=BLOCK_ORDER_DESIGN)
 return train.seal(value)

def block_order_validate_checkpoint(checkpoint,plan,binding,cursor):
 block_order_validate_schedule(plan['schedule'],plan['metadata']['original']);trajectory=block_order_trajectory(plan['metadata']['original'],plan['schedule'])
 need(checkpoint['trajectory']==trajectory and checkpoint['producer']['original_plan']==binding and checkpoint['position']==train.position(plan['schedule'],cursor)and len(checkpoint['steps'])==cursor,'BLOCK_ORDER_CHECKPOINT_TRAJECTORY_CURSOR')
 for j,step in enumerate(checkpoint['steps']):need(step.get('order_binding')==block_order_step_binding(plan['schedule'],j)and step.get('logical_step')==j+1 and step.get('skip_update')is False,'BLOCK_ORDER_CHECKPOINT_PREFIX')
 return block_order_cursor(plan['schedule'],cursor)

def block_order_strict_restore(model,optimizer,checkpoint,plan,binding,cursor,live,proof):
 block_order_validate_checkpoint(checkpoint,plan,binding,cursor)
 return train.strict_restore(model,optimizer,checkpoint,trajectory=plan['metadata']['trajectory'],original_plan_ref=binding,expected_cursor=cursor,current_runtime=live,expected_state_proof=proof)

def block_order_initial_reuse_metadata(original,lineage):
 score=read(lineage['original_INITIAL_score']);cold=read(original['inputs']['development_cold_result'])
 need(score['phase']=='INITIAL'and score['rows']==original['selection']['rows']and score['logits']==lineage['original_INITIAL_logits']and score['canonical_scores']==score['constant_scores']['factory'],'BLOCK_ORDER_ACTUAL_FACTORY_INITIAL')
 need(cold['status']=='COMPLETE_V3_DEVELOPMENT_COLD'and all(cold['bookends'].values())and cold['report']['cold_replay_bit_exact']is True and len(cold['report']['full_cursor3_proof'])==10 and cold['report']['technical_weights_never_fit_initializer']is True,'BLOCK_ORDER_REUSED_NUMERIC_COLD_SCOPE')
 return {'INITIAL_score':lineage['original_INITIAL_score'],'INITIAL_logits':lineage['original_INITIAL_logits'],'truth_sha256':score['truth_sha256'],'model_contract_sha256':original['model_contract']['contract_sha256'],'factory_model_sha256':original['initial_reference']['model_state_sha256'],'factory_full_RNG_sha256':original['initial_reference']['rng_sha256'],'old_numeric_cold_result':original['inputs']['development_cold_result'],'same_function_initial_outputs_reused':True,'new_order_cold_resume_parity':'NOT_MEASURED'}

def build_v3_direction_block_order_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_v3_direction_block_order_source_bridge
 original,c1,base,lineage,identity=recovery_original_metadata();design=read(BLOCK_ORDER_DESIGN);need(design['name']==BLOCK_ORDER_NAME and design['original_plan']==DEV_RECOVERY_ORIGINAL and design['original_INITIAL']==lineage['original_INITIAL_score'],'BLOCK_ORDER_DESIGN_ORIGINAL')
 reuse=block_order_initial_reuse_metadata(original,lineage);schedule=block_order_schedule(original);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'BLOCK_ORDER_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(base['runtime_compatibility'],runtime)
 owners=dict(original['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,d in owners.items():need(ref(REPO/n)['sha256']==d,'BLOCK_ORDER_SOURCE:'+n)
 inputs={**copy.deepcopy(original['inputs']),**{k:v for k,v in lineage.items()if k!='original_checkpoint'},'block_order_design':BLOCK_ORDER_DESIGN,'original_locked_PASS2':design['original_locked_PASS2'],'direction_diagnostic_review':design['diagnostic_review']}
 metadata_roles={k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt')and not k.startswith('provider_raw_response_')};payload=set(inputs)-metadata_roles;old={n:hashlib.sha256(subprocess.check_output(['git','show',BLOCK_ORDER_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 limits=development_limits('fit');eval_batches=(original['selection']['rows']+original['recipe']['eval_batch_rows']-1)//original['recipe']['eval_batch_rows'];diag=train.diagnostic_panels(original['schedule']);forwards=schedule['optimizer_updates']+original['recipe']['passes']*eval_batches+diag['supported_panels'];limits.update(model_forwards=forwards,backbone_forwards=forwards,readout_forwards=forwards,logit_archive_writes=original['recipe']['passes'],logit_archive_readbacks=original['recipe']['passes'])
 need(forwards==597 and diag['candidate_batch_indices']==[0,107,214]and not diag['skipped_empty_batch_indices'],'BLOCK_ORDER_DERIVED_BUDGET')
 return {'schema_version':'gx1_native_v3_direction_block_order_plan_v1','stage':BLOCK_ORDER_STAGE,'source_commit':head,'output_root':str(BLOCK_ORDER_ROOT),'source_owners':owners,'guard_source_bridge':native_v3_direction_block_order_source_bridge(old,producer_commit=BLOCK_ORDER_ANCHOR),'owner_source_bridge':block_order_source_bridge(),'design':BLOCK_ORDER_DESIGN,'original_plan':DEV_RECOVERY_ORIGINAL,'metadata':{'original':original,'trajectory':block_order_trajectory(original,schedule),'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(metadata_roles),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'physical_input_identity':identity,'initial_reuse':reuse,'model_contract':original['model_contract'],'initial_reference':original['initial_reference'],'recipe':original['recipe'],'schedule':schedule,'checkpoint_cursors':original['checkpoint_cursors'],'panel_indices':diag['candidate_batch_indices'],'limits':limits,'execution':development_execution('fit',original),'backbone_gradient_policy':None,'scope':design['candidate_and_qualification']}

def inspect_v3_direction_block_order(sha,admitted=False):
 binding={'path':str(BLOCK_ORDER_PLAN),'sha256':sha};plan=read(binding);need(plan==build_v3_direction_block_order_plan(plan['metadata']['runtime'],plan['source_commit']),'BLOCK_ORDER_EXACT_PLAN');recovery_check_inputs(plan);pref=ref(POLICY);p=read(pref);block_order_flags(p,admitted);parent=p.get(BLOCK_ORDER_STAGE,{})
 block_order_require_gate(parent,binding,admitted);need(parent.get('design')==BLOCK_ORDER_DESIGN,'BLOCK_ORDER_PARENT_DESIGN')
 if admitted:block_order_admission_snapshot(binding,pref)
 need(not BLOCK_ORDER_ROOT.exists()and not BLOCK_ORDER_ROOT.is_symlink(),'BLOCK_ORDER_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def block_order_physical(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 original=plan['metadata']['original'];schedule=block_order_validate_schedule(plan['schedule'],original);trajectory=block_order_trajectory(original,schedule);context=development_nn_context(original);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 need(context['truth_sha256']==plan['initial_reuse']['truth_sha256'],'BLOCK_ORDER_ACTUAL_INITIAL_TRUTH')
 for j,take in enumerate(context['batches']):
  need(len(take)==schedule['original_batch_rows'][j]and core.digest(context['source_rows'][take])==original['schedule']['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==original['schedule']['batch_issue_ns_sha256'][j],'BLOCK_ORDER_ACTUAL_WHOLE_BATCH')
 core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}});model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=original['initial_reference'];need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'BLOCK_ORDER_ACTUAL_FRESH_FACTORY_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'BLOCK_ORDER_FRESH_ADAMW')
 initial_reuse=development_publish(root,'INITIAL_REUSE.json',{'metadata_proof':plan['initial_reuse'],'fresh_model_sha256':core.state_sha(model.state_dict()),'fresh_full_RNG_sha256':core.state_sha(core.training_rng()),'same_named_INITIAL_outputs_reused':True,'new_INITIAL_forwards':0,'old_weight_checkpoint_loads':0});steps=[];scores={'INITIAL':plan['initial_reuse']['INITIAL_score']};receipts={};order_cursors={}
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'BLOCK_ORDER_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def checkpoint(cursor,name):
  cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=binding,lane='block_order',cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);order_cursors[str(cursor)]=block_order_validate_checkpoint(cp,plan,binding,cursor);receipt=development_checkpoint(root,name,cp,counts,binding);development_publish(root,Path(name).stem+'_ORDER.json',{'checkpoint_receipt':receipt,'order_cursor':order_cursors[str(cursor)]});receipts[str(cursor)]=receipt
 try:
  train.release_phase_allocator_cache(model,optimizer,label='DEV_INITIAL_SCORE_TO_TRAIN')
  for j,original_batch in enumerate(schedule['logical_original_batch_ids']):
   current=adapter.packet(context,context['batches'][original_batch]);step=train.training_step(model,optimizer,current,original,counts,j);need(not step['skip_update']and counts['optimizer_steps']==j+1,'BLOCK_ORDER_CANONICAL_UPDATE');step['order_binding']=block_order_step_binding(schedule,j);steps.append(step)
   if j+1==plan['checkpoint_cursors'][0]:checkpoint(j+1,'CURSOR001.pt')
   if(j+1)%schedule['batches_per_pass']==0:
    label='PASS'+str((j+1)//schedule['batches_per_pass']);train.release_phase_allocator_cache(model,optimizer,label='DEV_TRAIN_TO_'+label);scores[label]=development_score(model,context,original,root,label,counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_'+label+'_TO_NEXT_PHASE')
  before=core.state_sha(optimizer.state_dict());diagnostics=[]
  for index in plan['panel_indices']:diagnostics.append({'batch_index':index,'report':core.future_cuda_gradient_probe(model,adapter.packet(context,context['batches'][index]),original,counts)})
  need(core.state_sha(optimizer.state_dict())==before,'BLOCK_ORDER_DIAGNOSTICS_NO_OPTIMIZER_MUTATION');checkpoint(schedule['logical_batches'],'FINAL.pt')
  need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'BLOCK_ORDER_EXACT_COUNTS_VRAM')
  return {'counts':counts,'trajectory':trajectory,'schedule_sha256':schedule['contract_sha256'],'initial_reuse':initial_reuse,'scores':scores,'checkpoint_receipts':receipts,'checkpoint_order_cursors':order_cursors,'gradient_diagnostics':diagnostics,'optimizer_updates':counts['optimizer_steps'],'candidate_phase':'PASS2','new_order_cold_resume_parity':'NOT_MEASURED','later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
 finally:
  for hook in hooks:hook.remove()

def block_order_bookends(plan,binding,pref,boot):
 tests={'plan':lambda:ref(BLOCK_ORDER_PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==d for n,d in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and block_order_admission_snapshot(binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};checks={}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def run_v3_direction_block_order(sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_direction_block_order_execution
 execution=require_native_v3_direction_block_order_execution(plan_sha256=sha);plan,binding,pref,live=inspect_v3_direction_block_order(sha,True);root=BLOCK_ORDER_ROOT;root.mkdir(exist_ok=False);boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':block_order_admission_snapshot(binding,pref),'execution':execution,'boots':boot,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('BLOCK_ORDER_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed={str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.update(str(root/n)for n in('CURSOR001.pt','FINAL.pt','PASS1_LOGITS.npz','PASS2_LOGITS.npz'));sys.addaudithook(lambda e,a:development_io_audit(allowed,e,a));recovery_check_inputs(plan,physical=True);report=block_order_physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'BLOCK_ORDER_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=block_order_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'BLOCK_ORDER_BOOKEND_FAILED')
  result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_DIRECTION_BLOCK_ORDER'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal}

def main_v3_direction_block_order():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_direction_block_order_execution
 p=argparse.ArgumentParser();p.add_argument('--v3-direction-block-order-plan-sha256',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--v3-direction-block-order-plan-sha256',a.v3_direction_block_order_plan_sha256]+(['--preflight']if a.preflight else[]),'BLOCK_ORDER_EXACT_CLI')
 if a.preflight:
  require_native_v3_direction_block_order_execution(plan_sha256=a.v3_direction_block_order_plan_sha256);plan,binding,pref,live=inspect_v3_direction_block_order(a.v3_direction_block_order_plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_V3_DIRECTION_BLOCK_ORDER_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'initial_reuse':plan['initial_reuse'],'schedule_sha256':plan['schedule']['contract_sha256'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_direction_block_order(a.v3_direction_block_order_plan_sha256),sort_keys=True))

# BEGIN GX1 V3 BLOCK ORDER RECOVERY 20261128
BLOCK_RECOVERY_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001'
BLOCK_RECOVERY_STAGE=BLOCK_RECOVERY_NAME.lower()
BLOCK_RECOVERY_PROTOCOL=REPO/'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_PROTOCOL_20261128_001.json'
BLOCK_RECOVERY_LANES=('reference','cold','recovery')
BLOCK_RECOVERY_PLANS={lane:REPO/'configs/research'/(BLOCK_RECOVERY_NAME+'_'+lane.upper()+'.json')for lane in BLOCK_RECOVERY_LANES}
BLOCK_RECOVERY_ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(BLOCK_RECOVERY_NAME+'_'+lane)for lane in BLOCK_RECOVERY_LANES}
BLOCK_RECOVERY_DESIGN={'path':str(E/'V3_BLOCK_ORDER_EXACT_RECOVERY_DESIGN_20261128_001.json'),'sha256':'737dd510ca83958d1f1bb192ef22ca40e072c0aa9b5c757cc3787bb464ebb830'}
BLOCK_RECOVERY_ANCHOR='d405fbebdcd0621501def940bd90855bd1c5709f'
BLOCK_RECOVERY_DISPATCH=b" if sys.argv[1:2]==['--v3-block-order-recovery-plan-sha256']:return main_v3_block_order_recovery()\n"
BLOCK_RECOVERY_LEGACY_REVERSE=b" if BLOCK_RECOVERY_DISPATCH in first:\n  need(first.count(BLOCK_RECOVERY_DISPATCH)==1,'BLOCK_RECOVERY_DISPATCH');first=first.replace(BLOCK_RECOVERY_DISPATCH,b'')\n"
BLOCK_RECOVERY_RAW_REVERSE=b" if BLOCK_RECOVERY_DISPATCH in raw:\n  need(raw.count(BLOCK_RECOVERY_DISPATCH)==raw.count(BLOCK_RECOVERY_LEGACY_REVERSE)==1,'BLOCK_RECOVERY_OUTER_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'')\n"
BLOCK_RECOVERY_DIRECTION_REVERSE=b" if BLOCK_RECOVERY_DISPATCH in raw:\n  need(raw.count(BLOCK_RECOVERY_DISPATCH)==raw.count(BLOCK_RECOVERY_LEGACY_REVERSE)==raw.count(BLOCK_RECOVERY_RAW_REVERSE)==1,'BLOCK_RECOVERY_DIRECTION_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'').replace(BLOCK_RECOVERY_RAW_REVERSE,b'')\n"
BLOCK_RECOVERY_ORDER_REVERSE=b" if BLOCK_RECOVERY_DISPATCH in raw:\n  need(all(raw.count(f)==1 for f in (BLOCK_RECOVERY_DISPATCH,BLOCK_RECOVERY_LEGACY_REVERSE,BLOCK_RECOVERY_RAW_REVERSE,BLOCK_RECOVERY_DIRECTION_REVERSE)),'BLOCK_RECOVERY_ORDER_REVERSAL');raw=raw.replace(BLOCK_RECOVERY_DISPATCH,b'').replace(BLOCK_RECOVERY_LEGACY_REVERSE,b'').replace(BLOCK_RECOVERY_RAW_REVERSE,b'').replace(BLOCK_RECOVERY_DIRECTION_REVERSE,b'')\n"

def block_recovery_source_bridge():
 raw=(REPO/SELF).read_bytes();a=b'\n# BEGIN GX1 V3 BLOCK ORDER RECOVERY 20261128\n';b=b'# END GX1 V3 BLOCK ORDER RECOVERY 20261128\n';need(raw.count(a)==raw.count(b)==1,'BLOCK_RECOVERY_ONE_BLOCK');first,tail=raw.split(a);_,last=tail.split(b);raw=first+last
 raw=issue_mix_reverse_legacy(raw)
 for fragment in (BLOCK_RECOVERY_DISPATCH,BLOCK_RECOVERY_LEGACY_REVERSE,BLOCK_RECOVERY_RAW_REVERSE,BLOCK_RECOVERY_DIRECTION_REVERSE,BLOCK_RECOVERY_ORDER_REVERSE):
  need(raw.count(fragment)==1,'BLOCK_RECOVERY_EXACT_NORMALIZATION');raw=raw.replace(fragment,b'')
 old=subprocess.check_output(['git','show',BLOCK_RECOVERY_ANCHOR+':'+SELF],cwd=REPO);need(raw==old,'BLOCK_RECOVERY_OLD_LEAF_UNCHANGED')
 return {'producer_commit':BLOCK_RECOVERY_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'all_old_numeric_order_and_recovery_paths_unchanged':True,'only_additive_adapter_dispatch_and_exact_bridge_normalization':True}

def block_recovery_gate(binding,lane,admitted=False):
 need(lane in BLOCK_RECOVERY_LANES,'BLOCK_RECOVERY_LANE')
 return {'status':'ADMITTED_EXACT_V3_BLOCK_ORDER_RECOVERY_'+lane.upper()if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False}

def block_recovery_require_gate(value,binding,lane,admitted=False):
 expected=block_recovery_gate(binding,lane,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized'))and value==expected,'BLOCK_RECOVERY_EXACT_GATE');return value

def block_recovery_snapshot_path(lane):
 need(lane in BLOCK_RECOVERY_LANES,'BLOCK_RECOVERY_LANE');return E/('V3_BLOCK_ORDER_RECOVERY_'+lane.upper()+'_ADMITTED_POLICY_20261128_001.json')

def block_recovery_admission_snapshot(lane,binding,protocol,pref):
 sr=ref(block_recovery_snapshot_path(lane));p=read(sr);need(pref=={'path':str(POLICY),'sha256':sr['sha256']},'BLOCK_RECOVERY_SNAPSHOT_SHA');block_order_flags(p,True);parent=p.get(BLOCK_RECOVERY_STAGE,{})
 block_recovery_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,True)
 need(parent.get('execution_authorized')is True and parent.get('active_lane')==lane and parent.get('protocol')==protocol and parent.get('design')==BLOCK_RECOVERY_DESIGN and all(parent.get(k)is False for k in('main_fit_authorized','test_authorized','spending_authorized')),'BLOCK_RECOVERY_ADMITTED_SCOPE');return sr

def block_recovery_original_metadata():
 design=read(BLOCK_RECOVERY_DESIGN);binding=design['original_plan'];original=read(binding);receipt=read(design['original_C1_receipt']);order=read(design['original_C1_order']);root=Path(original['output_root']);cr=ref(root/'CLAIM.json');rr=ref(root/'RESULT.json');tr=ref(root/'TERMINAL.json');claim=read(cr);result=read(rr);terminal=read(tr)
 need(binding=={'path':str(BLOCK_ORDER_PLAN),'sha256':'15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008'}and design['experiment_id']==BLOCK_RECOVERY_NAME and original['schema_version']=='gx1_native_v3_direction_block_order_plan_v1','BLOCK_RECOVERY_EXACT_ORIGINAL')
 need(claim['plan']==result['plan']==terminal['plan']==binding and result['claim']==terminal['claim']==cr and terminal['result']==rr and result['status']==terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'BLOCK_RECOVERY_ORIGINAL_CHAIN')
 snapshot=block_order_admission_snapshot(binding,claim['policy_at_claim']);need(claim['admitted_policy_snapshot']==snapshot and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False,'BLOCK_RECOVERY_ORIGINAL_SNAPSHOT');core.future_cpu_parent_stopped(claim['process_identity'])
 nn=original['metadata']['original'];schedule=block_order_validate_schedule(original['schedule'],nn);trajectory=block_order_trajectory(nn,schedule);proof=train.require_recovery_checkpoint_proof(receipt['proof'])
 need(trajectory==original['metadata']['trajectory']and trajectory['contract_sha256']==design['original_trajectory_contract_sha256']and schedule['contract_sha256']==design['original_schedule_sha256'],'BLOCK_RECOVERY_FROZEN_TRAJECTORY')
 need(receipt['original_plan']==receipt['execution_plan']==binding and receipt['producer_lane']=='block_order'and type(receipt['logical_cursor'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==1 and receipt['checkpoint']==design['original_C1_checkpoint']and receipt['trajectory_sha256']==trajectory['contract_sha256'],'BLOCK_RECOVERY_AUTHENTIC_C1')
 need(order=={'checkpoint_receipt':design['original_C1_receipt'],'order_cursor':block_order_cursor(schedule,1)}and schedule['logical_original_batch_ids'][:2]==[116,131],'BLOCK_RECOVERY_NEXT_BATCH')
 need(proof['trajectory']==core.state_sha(trajectory)and proof['position']==core.state_sha(train.position(schedule,1))and proof['scores']==core.state_sha({'INITIAL':design['original_INITIAL']})and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'BLOCK_RECOVERY_C1_PROOF')
 core.future_cuda_runtime_matches(original['metadata']['runtime'],receipt['producer_runtime']);need(claim['boots']=={k:receipt['producer_runtime'][k]for k in('physical_windows_boot_id','wsl_boot_id')},'BLOCK_RECOVERY_ORIGINAL_BOOT')
 old,c1,base,lineage,identity=recovery_original_metadata();need(old==nn,'BLOCK_RECOVERY_UNCHANGED_NN_CONTEXT');reuse=block_order_initial_reuse_metadata(nn,lineage);ir=ref(root/'INITIAL_REUSE.json');initial=read(ir)
 need(initial['metadata_proof']==original['initial_reuse']==reuse and initial['fresh_model_sha256']==nn['initial_reference']['model_state_sha256']and initial['fresh_full_RNG_sha256']==nn['initial_reference']['rng_sha256']and initial['old_weight_checkpoint_loads']==initial['new_INITIAL_forwards']==0,'BLOCK_RECOVERY_ORIGINAL_FACTORY')
 refs={'original_plan':binding,'original_claim':cr,'original_result':rr,'original_terminal':tr,'original_admitted_policy':snapshot,'checkpoint_receipt':design['original_C1_receipt'],'original_checkpoint':design['original_C1_checkpoint'],'original_order_cursor':design['original_C1_order'],'original_INITIAL_reuse':ir,'original_INITIAL_score':design['original_INITIAL'],'original_stop_review':design['stop_review'],'original_maintenance_diagnosis':design['maintenance_diagnosis'],'block_recovery_design':BLOCK_RECOVERY_DESIGN};read(design['stop_review'])
 return original,receipt,trajectory,refs,identity

def build_v3_block_order_recovery_protocol():
 original,receipt,trajectory,refs,identity=block_recovery_original_metadata()
 return {'schema_version':'gx1_v3_block_order_recovery_protocol_v1','experiment_id':BLOCK_RECOVERY_NAME,'design':BLOCK_RECOVERY_DESIGN,'original_plan':refs['original_plan'],'checkpoint_receipt':refs['checkpoint_receipt'],'original_trajectory':trajectory,'original_schedule':original['schedule'],'original_lineage':refs,'physical_input_identity':identity,'limits':{lane:recovery_limits(lane,original['metadata']['original'])for lane in BLOCK_RECOVERY_LANES},'recipe':original['recipe'],'scope':{'original_logical_updates':432,'previous_physical_updates':'UNKNOWN_1_TO_216','technical_weights_never_initializer':True,'TEST_VAL_M1_GC_trade_spending':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}}

def block_recovery_closed(lane):
 root=BLOCK_RECOVERY_ROOTS[lane];binding=ref(BLOCK_RECOVERY_PLANS[lane]);tr=ref(root/'TERMINAL.json');terminal=read(tr);rr=terminal['result'];result=read(rr);claim=read(result['claim']);plan=read(binding)
 need(terminal['plan']==result['plan']==binding and terminal['claim']==result['claim']and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and result['status']=='COMPLETE_V3_BLOCK_ORDER_RECOVERY_'+lane.upper()and result['lane']==lane and result['error']is None and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'BLOCK_RECOVERY_CLOSED_'+lane.upper())
 need(claim['plan']==binding and claim['lane']==lane and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False,'BLOCK_RECOVERY_PARENT_CLAIM');snapshot=block_recovery_admission_snapshot(lane,binding,plan['protocol'],claim['policy_at_claim']);need(claim['admitted_policy_snapshot']==snapshot,'BLOCK_RECOVERY_PARENT_SNAPSHOT');core.future_cpu_parent_stopped(claim['process_identity'])
 return {'plan':binding,'result':rr,'terminal':tr,'claim':result['claim'],'admitted_policy_snapshot':snapshot,'value':result}

def build_v3_block_order_recovery_plan(lane,runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_v3_block_order_recovery_source_bridge
 need(lane in BLOCK_RECOVERY_LANES,'BLOCK_RECOVERY_LANE');pr=ref(BLOCK_RECOVERY_PROTOCOL);protocol=read(pr);need(protocol==build_v3_block_order_recovery_protocol(),'BLOCK_RECOVERY_EXACT_PROTOCOL')
 original=read(protocol['original_plan']);receipt=read(protocol['checkpoint_receipt']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'BLOCK_RECOVERY_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(protocol['original_trajectory']['runtime_compatibility'],runtime)
 inputs={**copy.deepcopy(original['inputs']),**protocol['original_lineage'],'block_recovery_protocol':pr};parents={}
 for parent_lane in (()if lane=='reference'else('reference',)if lane=='cold'else('reference','cold')):
  closed=block_recovery_closed(parent_lane);parent=read(closed['plan']);need(parent['protocol']==pr and parent['metadata']['trajectory']==protocol['original_trajectory']and parent['source_commit']==head,'BLOCK_RECOVERY_PARENT_GENERATION');parents[parent_lane]={k:closed[k]for k in('plan','result','terminal','claim','admitted_policy_snapshot')};inputs.update({parent_lane+'_'+k:closed[k]for k in parents[parent_lane]});report=closed['value']['report']
  if parent_lane=='reference':
   need(report['original_C1_all10_equal']is True,'BLOCK_RECOVERY_REFERENCE_C1');inputs['reference_cursor2_receipt']=report['cursor2_receipt'];r=read(report['cursor2_receipt']);need(r['original_plan']==r['execution_plan']==closed['plan']and r['producer_lane']=='reference'and r['logical_cursor']==r['optimizer_updates']==2 and r['trajectory_sha256']==protocol['original_trajectory']['contract_sha256'],'BLOCK_RECOVERY_REFERENCE_PRODUCER')
  else:need(report['cold_replay_bit_exact']is True and report['all10_proofs_equal']is True,'BLOCK_RECOVERY_COLD_PARITY')
 owners=dict(original['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,d in owners.items():need(ref(REPO/n)['sha256']==d,'BLOCK_RECOVERY_SOURCE:'+n)
 for parent_lane in parents:need(read(parents[parent_lane]['plan'])['source_owners']==owners,'BLOCK_RECOVERY_PARENT_SOURCE_FREEZE')
 mr={k for k,v in inputs.items()if Path(v['path']).suffix not in('.pt','.npz','.npy')and not k.startswith('provider_raw_response_')};payload=set(inputs)-mr;old={n:hashlib.sha256(subprocess.check_output(['git','show',BLOCK_RECOVERY_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 return {'schema_version':'gx1_native_v3_block_order_recovery_plan_v1','lane':lane,'stage':BLOCK_RECOVERY_STAGE,'output_root':str(BLOCK_RECOVERY_ROOTS[lane]),'source_commit':head,'source_owners':owners,'guard_source_bridge':native_v3_block_order_recovery_source_bridge(old,producer_commit=BLOCK_RECOVERY_ANCHOR),'owner_source_bridge':block_recovery_source_bridge(),'protocol':pr,'design':BLOCK_RECOVERY_DESIGN,'original_plan':protocol['original_plan'],'checkpoint_receipt':protocol['checkpoint_receipt'],'metadata':{'original':original,'receipt':receipt,'trajectory':protocol['original_trajectory'],'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(mr),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'physical_input_identity':protocol['physical_input_identity'],'parents':parents,'recipe':original['recipe'],'schedule':original['schedule'],'model_contract':original['model_contract'],'initial_reference':original['initial_reference'],'panel_indices':original['panel_indices'],'limits':protocol['limits'][lane],'execution':development_execution('fit'if lane=='recovery'else'warm',original),'starting_cursor':0 if lane=='reference'else 1,'ending_cursor':432 if lane=='recovery'else 2,'backbone_gradient_policy':None,'scope':protocol['scope']}

def inspect_v3_block_order_recovery(sha,lane,admitted=False):
 binding={'path':str(BLOCK_RECOVERY_PLANS[lane]),'sha256':sha};plan=read(binding);need(plan==build_v3_block_order_recovery_plan(lane,plan['metadata']['runtime'],plan['source_commit']),'BLOCK_RECOVERY_EXACT_PLAN');recovery_check_inputs(plan);pref=ref(POLICY);p=read(pref);block_order_flags(p,admitted);parent=p.get(BLOCK_RECOVERY_STAGE,{})
 block_recovery_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,admitted);need(parent.get('protocol')==plan['protocol']and parent.get('design')==BLOCK_RECOVERY_DESIGN,'BLOCK_RECOVERY_PARENT_SCOPE')
 if admitted:block_recovery_admission_snapshot(lane,binding,plan['protocol'],pref)
 need(not BLOCK_RECOVERY_ROOTS[lane].exists()and not BLOCK_RECOVERY_ROOTS[lane].is_symlink(),'BLOCK_RECOVERY_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def block_recovery_restore(model,optimizer,checkpoint,plan,live):
 original=plan['metadata']['original'];receipt=plan['metadata']['receipt'];binding=plan['original_plan'];need(checkpoint['producer']=={'original_plan':binding,'lane':'block_order','runtime':receipt['producer_runtime']}and checkpoint['scores']=={'INITIAL':plan['inputs']['original_INITIAL_score']},'BLOCK_RECOVERY_ACTUAL_PRODUCER_INITIAL')
 return block_order_strict_restore(model,optimizer,checkpoint,original,binding,1,live,receipt['proof'])

def block_recovery_physical(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 lane=plan['lane'];original=plan['metadata']['original'];nn=original['metadata']['original'];trajectory=plan['metadata']['trajectory'];schedule=block_order_validate_schedule(original['schedule'],nn);ids=schedule['logical_original_batch_ids'];context=development_nn_context(nn);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 need(context['truth_sha256']==original['initial_reuse']['truth_sha256'],'BLOCK_RECOVERY_INITIAL_TRUTH')
 for j,take in enumerate(context['batches']):need(len(take)==schedule['original_batch_rows'][j]and core.digest(context['source_rows'][take])==schedule['canonical_schedule']['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==schedule['canonical_schedule']['batch_issue_ns_sha256'][j],'BLOCK_RECOVERY_ACTUAL_BATCH')
 core.future_cuda_seed_and_runtime({'recipe':nn['recipe'],'metadata':{'runtime':live}});model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=nn['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'BLOCK_RECOVERY_FRESH_FACTORY_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'BLOCK_RECOVERY_FRESH_ADAMW');steps=[];scores={'INITIAL':plan['inputs']['original_INITIAL_score']};extra={}
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'BLOCK_RECOVERY_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def step(j):
  record=train.training_step(model,optimizer,adapter.packet(context,context['batches'][ids[j]]),nn,counts,j);need(record['skip_update']is False,'BLOCK_RECOVERY_NO_SKIP');record['order_binding']=block_order_step_binding(schedule,j);steps.append(record)
 def make(cursor):
  producer=binding if lane=='reference'else plan['original_plan'];cp=train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=producer,lane=lane,cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);block_order_validate_checkpoint(cp,original,producer,cursor);return cp
 try:
  if lane=='reference':
   train.release_phase_allocator_cache(model,optimizer,label='DEV_INITIAL_SCORE_TO_TRAIN');step(0);cp=make(1);proof=train.checkpoint_proof(cp);need(proof==plan['metadata']['receipt']['proof'],'BLOCK_RECOVERY_REFERENCE_C1_ALL10');extra['cursor1_proof']=development_publish(root,'REFERENCE_CURSOR001_PROOF.json',{'proof':proof,'actual_producer':cp['producer'],'execution_plan':binding,'original_plan':plan['original_plan'],'original_C1_receipt':plan['checkpoint_receipt'],'all10_equal':True});extra['original_C1_all10_equal']=True;del cp;step(1);cp=make(2);extra['cursor2_receipt']=development_checkpoint(root,'REFERENCE_CURSOR002.pt',cp,counts,binding);extra['full_cursor2_proof']=train.checkpoint_proof(cp)
  else:
   refcp=plan['inputs']['original_checkpoint'];need(ref(refcp['path'])==refcp,'BLOCK_RECOVERY_C1_SHA_BEFORE_LOAD');cp=torch.load(refcp['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1;position=block_recovery_restore(model,optimizer,cp,plan,live);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp;extra['restore_proof']=development_publish(root,'RESTORE001_PROOF.json',{'original_plan':plan['original_plan'],'execution_plan':binding,'checkpoint_receipt':plan['checkpoint_receipt'],'position':position,'order_cursor':block_order_cursor(schedule,1),'expected_state_proof':plan['metadata']['receipt']['proof'],'actual_producer_enforced':True,'canonical_strict_restore_completed':True,'new_model_forwards':0})
   if lane=='cold':
    step(1);cp=make(2);proof=train.checkpoint_proof(cp);need(proof==read(plan['inputs']['reference_cursor2_receipt'])['proof']and len(proof)==10,'BLOCK_RECOVERY_COLD_C2_ALL10');extra.update(cold_replay_bit_exact=True,all10_proofs_equal=True,full_cursor2_proof=proof,cursor2_receipt=development_checkpoint(root,'REPLAYED_CURSOR002.pt',cp,counts,binding))
   else:
    for j in range(1,schedule['logical_batches']):
     step(j);need(counts['optimizer_steps']+1==j+1 and len(steps)==j+1,'BLOCK_RECOVERY_GLOBAL_CURSOR')
     if(j+1)%schedule['batches_per_pass']==0:
      label='PASS'+str((j+1)//schedule['batches_per_pass']);train.release_phase_allocator_cache(model,optimizer,label='DEV_TRAIN_TO_'+label);scores[label]=development_score(model,context,nn,root,label,counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_'+label+'_TO_NEXT_PHASE')
    before=core.state_sha(optimizer.state_dict());extra['gradient_diagnostics']=[]
    for index in original['panel_indices']:extra['gradient_diagnostics'].append({'batch_index':index,'report':core.future_cuda_gradient_probe(model,adapter.packet(context,context['batches'][index]),nn,counts)})
    need(core.state_sha(optimizer.state_dict())==before,'BLOCK_RECOVERY_DIAGNOSTICS_NO_MUTATION');cp=make(432);extra['final_checkpoint_receipt']=development_checkpoint(root,'FINAL.pt',cp,counts,binding);extra['final_order_cursor']=block_order_cursor(schedule,432)
  need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'BLOCK_RECOVERY_EXACT_COUNTS_VRAM')
  return {'counts':counts,'original_plan':plan['original_plan'],'original_checkpoint_receipt':plan['checkpoint_receipt'],'execution_plan':binding,'trajectory':trajectory,'scores':scores,'starting_cursor':plan['starting_cursor'],'ending_cursor':plan['ending_cursor'],'new_optimizer_updates':counts['optimizer_steps'],'logical_original_optimizer_updates':plan['ending_cursor'],'previous_physical_updates':'UNKNOWN_1_TO_216','technical_weights_never_initializer':True,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN',**extra}
 finally:
  for hook in hooks:hook.remove()

def block_recovery_bookends(plan,binding,pref,boot):
 tests={'plan':lambda:ref(BLOCK_RECOVERY_PLANS[plan['lane']])==binding,'sources':lambda:all(ref(REPO/n)['sha256']==d for n,d in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and block_recovery_admission_snapshot(plan['lane'],binding,plan['protocol'],pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};checks={}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def run_v3_block_order_recovery(sha,lane):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_block_order_recovery_execution
 execution=require_native_v3_block_order_recovery_execution(plan_sha256=sha,lane=lane);plan,binding,pref,live=inspect_v3_block_order_recovery(sha,lane,True);root=BLOCK_RECOVERY_ROOTS[lane];root.mkdir(exist_ok=False);boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':block_recovery_admission_snapshot(lane,binding,plan['protocol'],pref),'lane':lane,'original_plan':plan['original_plan'],'checkpoint_receipt':plan['checkpoint_receipt'],'execution':execution,'boots':boot,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('BLOCK_RECOVERY_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed={str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.add(str(root/('REFERENCE_CURSOR002.pt'if lane=='reference'else'REPLAYED_CURSOR002.pt'if lane=='cold'else'FINAL.pt')))
  if lane=='recovery':allowed.update(str(root/(label+'_LOGITS.npz'))for label in('PASS1','PASS2'))
  sys.addaudithook(lambda e,a:development_io_audit(allowed,e,a));recovery_check_inputs(plan,physical=True);report=block_recovery_physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'BLOCK_RECOVERY_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=block_recovery_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'BLOCK_RECOVERY_BOOKEND_FAILED')
  result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_BLOCK_ORDER_RECOVERY_'+lane.upper()if error is None else'FAILED_CONSUMED','lane':lane,'plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal,'lane':lane}

def main_v3_block_order_recovery():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_block_order_recovery_execution
 p=argparse.ArgumentParser();p.add_argument('--v3-block-order-recovery-plan-sha256',required=True);p.add_argument('--lane',choices=BLOCK_RECOVERY_LANES,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--v3-block-order-recovery-plan-sha256',a.v3_block_order_recovery_plan_sha256,'--lane',a.lane]+(['--preflight']if a.preflight else[]),'BLOCK_RECOVERY_EXACT_CLI')
 if a.preflight:
  require_native_v3_block_order_recovery_execution(plan_sha256=a.v3_block_order_recovery_plan_sha256,lane=a.lane);plan,binding,pref,live=inspect_v3_block_order_recovery(a.v3_block_order_recovery_plan_sha256,a.lane,False);print(json.dumps({'status':'PREFLIGHT_V3_BLOCK_ORDER_RECOVERY_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'original_plan':plan['original_plan'],'checkpoint_receipt':plan['checkpoint_receipt'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_block_order_recovery(a.v3_block_order_recovery_plan_sha256,a.lane),sort_keys=True))
# BEGIN GX1 V3 DIRECTION ISSUE MIX 20261128
ISSUE_MIX_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001'
ISSUE_MIX_STAGE=ISSUE_MIX_NAME.lower()
ISSUE_MIX_DESIGN={'path':str(E/'V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json'),'sha256':'93a59021614b48f67a0b71e66c29926e3fc302d2f4bcf6187e9f8bdf7c0d75d2'}
ISSUE_MIX_PROTOCOL=REPO/'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_PROTOCOL_20261128_001.json'
ISSUE_MIX_LANES=('reference','cold','fit')
ISSUE_MIX_PLANS={lane:REPO/'configs/research'/(ISSUE_MIX_NAME+'_'+lane.upper()+'.json')for lane in ISSUE_MIX_LANES}
ISSUE_MIX_ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(ISSUE_MIX_NAME+'_'+lane)for lane in ISSUE_MIX_LANES}
ISSUE_MIX_PROFILE='gx1_native_provider_hlc_v3_issue_mix_trainer20g_v1'
ISSUE_MIX_ANCHOR='4cb788b15fb47f2b6b2273d34485a34753649f72'
ISSUE_MIX_DISPATCH=b" if sys.argv[1:2]==['--v3-issue-mix-plan-sha256']:return main_v3_issue_mix()\n"

def issue_mix_reverse_legacy(raw):
 raw=train_eval_gap_reverse_legacy(raw)
 for fragment in (ISSUE_MIX_DISPATCH,b' first=issue_mix_reverse_legacy(first)\n',b' raw=issue_mix_reverse_legacy(raw)\n'):raw=raw.replace(fragment,b'')
 return raw

def issue_mix_source_bridge():
 raw=(REPO/SELF).read_bytes();start=b'# BEGIN GX1 V3 DIRECTION ISSUE MIX 20261128\n';end=b'# END GX1 V3 DIRECTION ISSUE MIX 20261128\n';need(raw.count(start)==raw.count(end)==1,'ISSUE_MIX_ONE_BLOCK');first,tail=raw.split(start);_,last=tail.split(end);raw=issue_mix_reverse_legacy(first+last);old=subprocess.check_output(['git','show',ISSUE_MIX_ANCHOR+':'+SELF],cwd=REPO);need(raw==old,'ISSUE_MIX_EXACT_ADDITIVE_REVERSAL')
 return {'producer_commit':ISSUE_MIX_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'all_old_numeric_context_scoring_and_restore_owners_unchanged':True,'only_additive_issue_mix_dispatch_and_exact_bridge_normalization':True}

def issue_mix_gate(binding,lane,admitted=False):
 need(lane in ISSUE_MIX_LANES,'ISSUE_MIX_LANE');return {'status':'ADMITTED_EXACT_V3_ISSUE_MIX_'+lane.upper()if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False}

def issue_mix_require_gate(value,binding,lane,admitted=False):
 expected=issue_mix_gate(binding,lane,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized'))and all(value.get(k)==v for k,v in expected.items()),'ISSUE_MIX_TYPED_ONE_USE_GATE');return value

def issue_mix_snapshot_path(lane):
 need(lane in ISSUE_MIX_LANES,'ISSUE_MIX_LANE');return E/('V3_DIRECTION_ISSUE_MIX_'+lane.upper()+'_ADMITTED_POLICY_20261128_001.json')

def issue_mix_admission_snapshot(lane,binding,protocol,pref):
 sr=ref(issue_mix_snapshot_path(lane));p=read(sr);need(pref=={'path':str(POLICY),'sha256':sr['sha256']},'ISSUE_MIX_SNAPSHOT_SHA');block_order_flags(p,True);parent=p.get(ISSUE_MIX_STAGE,{})
 issue_mix_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,True);need(parent.get('execution_authorized')is True and parent.get('active_lane')==lane and parent.get('protocol')==protocol and parent.get('design')==ISSUE_MIX_DESIGN and parent.get('original_plan')==read(ISSUE_MIX_DESIGN)['evidence']['original_plan']and all(parent.get(k)is False for k in('main_fit_authorized','test_authorized','spending_authorized')),'ISSUE_MIX_ADMITTED_SCOPE');return sr

def issue_mix_schedule(control,design):
 original=control['metadata']['original'];canonical=original['schedule'];ordered=block_order_validate_schedule(control['schedule'],original);rows=canonical['rows'];passes=original['recipe']['passes'];n=canonical['batches_per_pass'];sizes=[ordered['original_batch_rows'][i]for i in ordered['logical_original_batch_ids']];bound=design['ordering']
 need(rows==20727 and passes==2 and n==216 and sizes==bound['logical_step_rows']and ordered['contract_sha256']==bound['control_schedule_sha256']and design['fixed']['factory_seed']==original['recipe']['seed']==bound['seed']and bound['generator']=='private_numpy_Generator_PCG64','ISSUE_MIX_FIXED_GEOMETRY')
 private=np.random.Generator(np.random.PCG64(original['recipe']['seed']));takes=[];raw_hashes=[];stream_hashes=[]
 for p in range(passes):
  raw=private.permutation(rows).astype(np.int64);raw_hashes.append(core.state_sha(raw));offset=0;parts=[]
  for size in sizes[p*n:(p+1)*n]:
   take=np.sort(raw[offset:offset+size]);need(len(take)==size and np.all(np.diff(take)>0),'ISSUE_MIX_SORTED_MEMBERSHIP');takes.append(take.tolist());parts.append(take);offset+=size
  stream=np.concatenate(parts);need(offset==rows and np.array_equal(np.sort(stream),np.arange(rows,dtype=np.int64)),'ISSUE_MIX_PASS_BIJECTION');stream_hashes.append(core.state_sha(stream))
 hashes=[core.state_sha(np.asarray(t,dtype=np.int64))for t in takes]
 need(raw_hashes==bound['raw_permutation_state_sha256_by_pass']and stream_hashes==bound['actual_sorted_partition_stream_state_sha256_by_pass']and hashes==bound['actual_cohort_take_state_sha256_by_logical_batch'],'ISSUE_MIX_DESIGN_STREAM_HASHES')
 return train.seal({'schema_version':'gx1_native_v3_issue_mix_schedule_v1','rows':rows,'passes':passes,'batch_rows':original['recipe']['train_batch_rows'],'batches_per_pass':n,'logical_batches':n*passes,'optimizer_updates':n*passes,'skipped_batches':0,'update_flags':[True]*(n*passes),'drop_last':False,'shuffle':True,'generator':bound['generator'],'seed':original['recipe']['seed'],'control_schedule_sha256':ordered['contract_sha256'],'logical_step_rows':sizes,'raw_permutation_state_sha256_by_pass':raw_hashes,'actual_sorted_partition_stream_state_sha256_by_pass':stream_hashes,'cohort_takes':takes,'cohort_take_state_sha256_by_logical_batch':hashes,'source_rows_sha256':canonical['source_rows_sha256'],'issue_ns_sha256':canonical['issue_ns_sha256']})

def issue_mix_validate_schedule(schedule,control,design):
 need(type(schedule)is dict and type(schedule.get('cohort_takes'))is list and all(type(t)is list and all(type(i)is int for i in t)for t in schedule['cohort_takes']),'ISSUE_MIX_TYPED_TAKES');need(schedule==issue_mix_schedule(control,design),'ISSUE_MIX_EXACT_SCHEDULE');return schedule

def issue_mix_trajectory(original,schedule,control_ref):
 value={k:copy.deepcopy(v)for k,v in development_trajectory(original).items()if k!='contract_sha256'};value.update(schema_version='gx1_v3_issue_mix_training_trajectory_v1',schedule=schedule,issue_mix_design=ISSUE_MIX_DESIGN,control_plan=control_ref,cohort_prior=original['inputs']['cohort_prior'],cohort_selection=original['selection']);return train.seal(value)

def issue_mix_limits(lane,original):
 need(lane in ISSUE_MIX_LANES,'ISSUE_MIX_LANE')
 if lane=='fit':
  limits=development_limits('fit');e=(original['selection']['rows']+original['recipe']['eval_batch_rows']-1)//original['recipe']['eval_batch_rows'];f=original['schedule']['optimizer_updates']+original['recipe']['passes']*e+len(train.diagnostic_panels(original['schedule'])['candidate_batch_indices']);need(f==597,'ISSUE_MIX_DERIVED_FORWARD_BUDGET');limits.update(model_forwards=f,backbone_forwards=f,readout_forwards=f,logit_archive_writes=original['recipe']['passes'],logit_archive_readbacks=original['recipe']['passes']);return limits
 limits=recovery_limits(lane,original)
 if lane=='reference':limits.update(checkpoint_writes=2,checkpoint_readbacks=2,checkpoint_receipt_writes=2,checkpoint_prepublication_loads=2)
 return limits

def issue_mix_original_metadata():
 design=read(ISSUE_MIX_DESIGN);control_ref=design['evidence']['original_plan'];control=read(control_ref);original,_,_,lineage,identity=recovery_original_metadata();need(control_ref=={'path':str(BLOCK_ORDER_PLAN),'sha256':'15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008'}and control['metadata']['original']==original,'ISSUE_MIX_AUTHENTIC_CONTROL_CONTEXT');reuse=block_order_initial_reuse_metadata(original,lineage);need(reuse==control['initial_reuse'],'ISSUE_MIX_UNCHANGED_INITIAL')
 review=read(design['evidence']['TRAIN_review']);need(review['candidate_phase']=='PASS2'and review['INITIAL']==reuse['INITIAL_score'],'ISSUE_MIX_LOCKED_COMPARISONS');refs={**{k:v for k,v in lineage.items()if k!='original_checkpoint'},'issue_mix_design':ISSUE_MIX_DESIGN,'control_plan':control_ref,'chronological_locked_PASS2':review['chronological_PASS2'],'whole_batch_locked_PASS2':review['new_order_PASS2'],**{'design_'+k:v for k,v in design['evidence'].items()}}
 return original,control,design,refs,identity,reuse

def build_v3_issue_mix_protocol():
 original,control,design,refs,identity,reuse=issue_mix_original_metadata();schedule=issue_mix_schedule(control,design)
 return {'schema_version':'gx1_v3_direction_issue_mix_protocol_v1','experiment_id':ISSUE_MIX_NAME,'design':ISSUE_MIX_DESIGN,'original_plan':design['evidence']['original_plan'],'original_schedule':control['schedule'],'original_trajectory':issue_mix_trajectory(original,schedule,design['evidence']['original_plan']),'schedule':schedule,'original_lineage':refs,'physical_input_identity':identity,'initial_reuse':reuse,'recipe':original['recipe'],'limits':{lane:issue_mix_limits(lane,original)for lane in ISSUE_MIX_LANES},'scope':{'single_changed_variable':design['single_changed_variable'],'candidate_phase':'PASS2','only_all5_and_original_stability_criteria':True,'technical_weights_never_initializer':True,'TEST_VAL_M1_GC_trade_spending':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}}

def issue_mix_closed(lane):
 root=ISSUE_MIX_ROOTS[lane];binding=ref(ISSUE_MIX_PLANS[lane]);tr=ref(root/'TERMINAL.json');terminal=read(tr);rr=terminal['result'];result=read(rr);claim=read(result['claim']);plan=read(binding)
 need(terminal['plan']==result['plan']==claim['plan']==binding and terminal['claim']==result['claim']and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False and result['status']=='COMPLETE_V3_ISSUE_MIX_'+lane.upper()and result['lane']==claim['lane']==lane and result['error']is None and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'ISSUE_MIX_CLOSED_'+lane.upper());snapshot=issue_mix_admission_snapshot(lane,binding,plan['protocol'],claim['policy_at_claim']);need(claim['admitted_policy_snapshot']==snapshot,'ISSUE_MIX_PARENT_SNAPSHOT');core.future_cpu_parent_stopped(claim['process_identity'])
 return {'plan':binding,'result':rr,'terminal':tr,'claim':result['claim'],'admitted_policy_snapshot':snapshot,'value':result}

def build_v3_issue_mix_plan(lane,runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_v3_issue_mix_source_bridge,_native_v3_issue_mix_execution
 need(lane in ISSUE_MIX_LANES,'ISSUE_MIX_LANE');pr=ref(ISSUE_MIX_PROTOCOL);protocol=read(pr);need(protocol==build_v3_issue_mix_protocol(),'ISSUE_MIX_EXACT_PROTOCOL');control=read(protocol['original_plan']);original=control['metadata']['original'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'ISSUE_MIX_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(protocol['original_trajectory']['runtime_compatibility'],runtime)
 inputs={**copy.deepcopy(original['inputs']),**protocol['original_lineage'],'issue_mix_protocol':pr};parents={}
 for parent_lane in (()if lane=='reference'else('reference',)if lane=='cold'else('reference','cold')):
  closed=issue_mix_closed(parent_lane);parent=read(closed['plan']);need(parent['protocol']==pr and parent['metadata']['trajectory']==protocol['original_trajectory']and parent['source_commit']==head,'ISSUE_MIX_PARENT_GENERATION');parents[parent_lane]={k:closed[k]for k in('plan','result','terminal','claim','admitted_policy_snapshot')};inputs.update({parent_lane+'_'+k:closed[k]for k in parents[parent_lane]});report=closed['value']['report'];inputs[parent_lane+'_stream_binding']=report['stream_binding']
  if parent_lane=='reference':
   for cursor in (1,2):
    role='reference_cursor'+str(cursor)+'_receipt';inputs[role]=report['cursor'+str(cursor)+'_receipt'];r=read(inputs[role]);need(r['original_plan']==r['execution_plan']==closed['plan']and r['producer_lane']=='reference'and r['logical_cursor']==r['optimizer_updates']==cursor and r['trajectory_sha256']==protocol['original_trajectory']['contract_sha256'],'ISSUE_MIX_REFERENCE_PRODUCER')
   inputs['reference_checkpoint']=read(inputs['reference_cursor1_receipt'])['checkpoint']
  else:need(report['cold_replay_bit_exact']is True and report['all10_proofs_equal']is True,'ISSUE_MIX_COLD_PARITY')
 owners=dict(original['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,d in owners.items():need(ref(REPO/n)['sha256']==d,'ISSUE_MIX_SOURCE:'+n)
 for parent_lane in parents:need(read(parents[parent_lane]['plan'])['source_owners']==owners,'ISSUE_MIX_PARENT_SOURCE_FREEZE')
 mr={k for k,v in inputs.items()if Path(v['path']).suffix not in('.pt','.npz','.npy')and not k.startswith('provider_raw_response_')};payload=set(inputs)-mr;old={n:hashlib.sha256(subprocess.check_output(['git','show',ISSUE_MIX_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 return {'schema_version':'gx1_native_v3_direction_issue_mix_plan_v1','lane':lane,'stage':ISSUE_MIX_STAGE,'output_root':str(ISSUE_MIX_ROOTS[lane]),'execution_profile':ISSUE_MIX_PROFILE,'source_commit':head,'source_owners':owners,'guard_source_bridge':native_v3_issue_mix_source_bridge(old,producer_commit=ISSUE_MIX_ANCHOR),'owner_source_bridge':issue_mix_source_bridge(),'protocol':pr,'design':ISSUE_MIX_DESIGN,'original_plan':protocol['original_plan'],'metadata':{'original':original,'control':control,'trajectory':protocol['original_trajectory'],'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(mr),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'physical_input_identity':protocol['physical_input_identity'],'parents':parents,'recipe':original['recipe'],'schedule':protocol['schedule'],'model_contract':original['model_contract'],'initial_reference':original['initial_reference'],'initial_reuse':protocol['initial_reuse'],'panel_indices':control['panel_indices'],'limits':protocol['limits'][lane],'execution':_native_v3_issue_mix_execution(lane),'starting_cursor':1 if lane=='cold'else 0,'ending_cursor':432 if lane=='fit'else 2,'checkpoint_cursors':[1,432]if lane=='fit'else[1,2]if lane=='reference'else[2],'backbone_gradient_policy':None,'scope':protocol['scope']}

def inspect_v3_issue_mix(sha,lane,admitted=False):
 need(lane in ISSUE_MIX_LANES,'ISSUE_MIX_LANE');binding={'path':str(ISSUE_MIX_PLANS[lane]),'sha256':sha};plan=read(binding);need(plan==build_v3_issue_mix_plan(lane,plan['metadata']['runtime'],plan['source_commit']),'ISSUE_MIX_EXACT_PLAN');recovery_check_inputs(plan);pref=ref(POLICY);p=read(pref);block_order_flags(p,admitted);parent=p.get(ISSUE_MIX_STAGE,{})
 issue_mix_require_gate(parent.get('lane_gates',{}).get(lane),binding,lane,admitted);need(parent.get('protocol')==plan['protocol']and parent.get('design')==ISSUE_MIX_DESIGN and parent.get('original_plan')==plan['original_plan'],'ISSUE_MIX_PARENT_SCOPE')
 if admitted:issue_mix_admission_snapshot(lane,binding,plan['protocol'],pref)
 need(not ISSUE_MIX_ROOTS[lane].exists()and not ISSUE_MIX_ROOTS[lane].is_symlink(),'ISSUE_MIX_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def issue_mix_actual_stream(context,plan):
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 schedule=plan['schedule'];original=plan['metadata']['original'];need(context['truth_sha256']==plan['initial_reuse']['truth_sha256']and context['schedule']==original['schedule']and core.digest(context['source_rows'])==schedule['source_rows_sha256']and core.digest(context['truth']['issue_ns'])==schedule['issue_ns_sha256'],'ISSUE_MIX_ACTUAL_COHORT_IDENTITY');records=[]
 for index,values in enumerate(schedule['cohort_takes']):
  take=np.asarray(values,dtype=np.int64);truth=train.slice_provider_hlc_truth(context['truth'],take,target_contract=context['target_contract'],fold_upper_ns=train.UPPER);masks=objective._native_future_state_truth(truth,rows=len(take),fold_upper_ns=train.UPPER,target_contract=context['target_contract']);need(len(take)==schedule['logical_step_rows'][index]and int(masks['activity'].sum())>0,'ISSUE_MIX_ACTUAL_SUPPORTED_BATCH')
  records.append({'logical_index':index,'rows':len(take),'cohort_take_state_sha256':core.state_sha(take),'source_rows_sha256':core.digest(context['source_rows'][take]),'issue_ns_sha256':core.digest(context['truth']['issue_ns'][take]),'truth_sha256':objective.native_future_state_truth_sha256(truth,target_contract=context['target_contract']),'known_activity_cells':int(masks['activity'].sum())})
 n=schedule['batches_per_pass'];streams=[np.concatenate([np.asarray(t,dtype=np.int64)for t in schedule['cohort_takes'][p*n:(p+1)*n]])for p in range(schedule['passes'])]
 return train.seal({'schema_version':'gx1_v3_issue_mix_physical_stream_v1','schedule_sha256':schedule['contract_sha256'],'cohort_truth_sha256':context['truth_sha256'],'batches':records,'source_stream_sha256_by_pass':[core.digest(context['source_rows'][s])for s in streams],'issue_stream_sha256_by_pass':[core.digest(context['truth']['issue_ns'][s])for s in streams],'ordered_truth_binding_sha256':core.state_sha([r['truth_sha256']for r in records])})

def issue_mix_step_binding(schedule,stream,index):
 need(type(index)is int and 0<=index<schedule['logical_batches']and stream['schedule_sha256']==schedule['contract_sha256'],'ISSUE_MIX_STEP_INDEX');return {'schedule_sha256':schedule['contract_sha256'],'stream_sha256':stream['contract_sha256'],'pass_index':index//schedule['batches_per_pass'],'offset_in_pass':index%schedule['batches_per_pass'],**stream['batches'][index]}

def issue_mix_cursor(schedule,stream,cursor):
 return {'position':train.position(schedule,cursor),'schedule_sha256':schedule['contract_sha256'],'stream_sha256':stream['contract_sha256'],'next_issue_batch':None if cursor==schedule['logical_batches']else issue_mix_step_binding(schedule,stream,cursor)}

def issue_mix_validate_checkpoint(checkpoint,plan,producer,cursor,stream):
 schedule=issue_mix_validate_schedule(plan['schedule'],plan['metadata']['control'],read(ISSUE_MIX_DESIGN));need(checkpoint['trajectory']==plan['metadata']['trajectory']and checkpoint['trajectory']==issue_mix_trajectory(plan['metadata']['original'],schedule,plan['original_plan'])and checkpoint['producer']['original_plan']==producer and checkpoint['position']==train.position(schedule,cursor)and len(checkpoint['steps'])==cursor,'ISSUE_MIX_CHECKPOINT_TRAJECTORY_CURSOR')
 for j,step in enumerate(checkpoint['steps']):need(step.get('issue_mix_binding')==issue_mix_step_binding(schedule,stream,j)and step.get('logical_step')==j+1 and step.get('skip_update')is False,'ISSUE_MIX_CHECKPOINT_PREFIX')
 return issue_mix_cursor(schedule,stream,cursor)

def issue_mix_restore(model,optimizer,checkpoint,plan,live,stream):
 receipt=read(plan['inputs']['reference_cursor1_receipt']);producer=plan['parents']['reference']['plan'];need(checkpoint['producer']=={'original_plan':producer,'lane':'reference','runtime':receipt['producer_runtime']}and checkpoint['scores']=={'INITIAL':plan['initial_reuse']['INITIAL_score']},'ISSUE_MIX_ACTUAL_REFERENCE_PRODUCER_INITIAL');issue_mix_validate_checkpoint(checkpoint,plan,producer,1,stream)
 return train.strict_restore(model,optimizer,checkpoint,trajectory=plan['metadata']['trajectory'],original_plan_ref=producer,expected_cursor=1,current_runtime=live,expected_state_proof=receipt['proof'])

def issue_mix_physical(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 lane=plan['lane'];original=plan['metadata']['original'];schedule=issue_mix_validate_schedule(plan['schedule'],plan['metadata']['control'],read(ISSUE_MIX_DESIGN));context=development_nn_context(original);stream=issue_mix_actual_stream(context,plan);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;extra={}
 for parent_lane in plan['parents']:need(read(plan['inputs'][parent_lane+'_stream_binding'])==stream,'ISSUE_MIX_PARENT_ACTUAL_STREAM')
 stream_ref=development_publish(root,'STREAM_BINDING.json',stream);core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}});model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=original['initial_reference'];need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'ISSUE_MIX_FRESH_FACTORY_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'ISSUE_MIX_FRESH_ADAMW');scores={'INITIAL':plan['initial_reuse']['INITIAL_score']};steps=[];receipts={};cursors={}
 extra['initial_reuse']=development_publish(root,'INITIAL_REUSE.json',{'metadata_proof':plan['initial_reuse'],'fresh_model_sha256':core.state_sha(model.state_dict()),'fresh_full_RNG_sha256':core.state_sha(core.training_rng()),'actual_truth_sha256':context['truth_sha256'],'same_named_INITIAL_outputs_reused':True,'new_INITIAL_forwards':0,'old_weight_checkpoint_loads':0})
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'ISSUE_MIX_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def make(cursor):
  producer=plan['parents']['reference']['plan']if lane=='cold'else binding;cp=train.make_checkpoint(model,optimizer,trajectory=plan['metadata']['trajectory'],original_plan_ref=producer,lane=lane,cursor=cursor,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);cursors[str(cursor)]=issue_mix_validate_checkpoint(cp,plan,producer,cursor,stream);return cp
 def checkpoint(cursor,name):
  cp=make(cursor);r=development_checkpoint(root,name,cp,counts,binding);development_publish(root,Path(name).stem+'_ORDER.json',{'checkpoint_receipt':r,'issue_mix_cursor':cursors[str(cursor)]});receipts[str(cursor)]=r;return r
 try:
  if lane=='cold':
   refcp=plan['inputs']['reference_checkpoint'];need(ref(refcp['path'])==refcp,'ISSUE_MIX_C1_SHA_BEFORE_LOAD');cp=torch.load(refcp['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1;issue_mix_restore(model,optimizer,cp,plan,live,stream);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
  train.release_phase_allocator_cache(model,optimizer,label='DEV_INITIAL_SCORE_TO_TRAIN')
  for j in range(plan['starting_cursor'],plan['ending_cursor']):
   take=np.asarray(schedule['cohort_takes'][j],dtype=np.int64);record=train.training_step(model,optimizer,adapter.packet(context,take),original,counts,j);need(record['skip_update']is False and counts['optimizer_steps']+plan['starting_cursor']==j+1,'ISSUE_MIX_EXACT_STEP');record['issue_mix_binding']=issue_mix_step_binding(schedule,stream,j);steps.append(record)
   if lane=='reference':extra['cursor'+str(j+1)+'_receipt']=checkpoint(j+1,'REFERENCE_CURSOR'+str(j+1).zfill(3)+'.pt')
   elif lane=='cold':
    cp=make(2);proof=train.checkpoint_proof(cp);need(proof==read(plan['inputs']['reference_cursor2_receipt'])['proof']and len(proof)==10,'ISSUE_MIX_COLD_C2_ALL10');extra.update(cold_replay_bit_exact=True,all10_proofs_equal=True,cursor2_receipt=checkpoint(2,'REPLAYED_CURSOR002.pt'),full_cursor2_proof=proof)
   else:
    if j+1==1:checkpoint(1,'CURSOR001.pt')
    if(j+1)%schedule['batches_per_pass']==0:
     label='PASS'+str((j+1)//schedule['batches_per_pass']);train.release_phase_allocator_cache(model,optimizer,label='DEV_TRAIN_TO_'+label);scores[label]=development_score(model,context,original,root,label,counts);train.release_phase_allocator_cache(model,optimizer,label='DEV_'+label+'_TO_NEXT_PHASE')
  if lane=='reference':extra['full_cursor2_proof']=train.checkpoint_proof(make(2))
  if lane=='fit':
   before=core.state_sha(optimizer.state_dict());extra['gradient_diagnostics']=[]
   for index in plan['panel_indices']:extra['gradient_diagnostics'].append({'batch_index':index,'report':core.future_cuda_gradient_probe(model,adapter.packet(context,context['batches'][index]),original,counts)})
   need(core.state_sha(optimizer.state_dict())==before,'ISSUE_MIX_DIAGNOSTICS_NO_OPTIMIZER_MUTATION');extra['final_checkpoint_receipt']=checkpoint(schedule['logical_batches'],'FINAL.pt')
  need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'ISSUE_MIX_EXACT_COUNTS_VRAM')
  return {'counts':counts,'stream_binding':stream_ref,'trajectory':plan['metadata']['trajectory'],'scores':scores,'checkpoint_receipts':receipts,'checkpoint_order_cursors':cursors,'starting_cursor':plan['starting_cursor'],'ending_cursor':plan['ending_cursor'],'new_optimizer_updates':counts['optimizer_steps'],'final_order_cursor':issue_mix_cursor(schedule,stream,plan['ending_cursor']),'technical_weights_never_initializer':True,'candidate_phase':'PASS2'if lane=='fit'else None,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN',**extra}
 finally:
  for hook in hooks:hook.remove()

def issue_mix_bookends(plan,binding,pref,boot):
 tests={'plan':lambda:ref(ISSUE_MIX_PLANS[plan['lane']])==binding,'sources':lambda:all(ref(REPO/n)['sha256']==d for n,d in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and issue_mix_admission_snapshot(plan['lane'],binding,plan['protocol'],pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};checks={}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def run_v3_issue_mix(sha,lane):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_issue_mix_execution
 execution=require_native_v3_issue_mix_execution(plan_sha256=sha,lane=lane);plan,binding,pref,live=inspect_v3_issue_mix(sha,lane,True);root=ISSUE_MIX_ROOTS[lane];root.mkdir(exist_ok=False);boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':issue_mix_admission_snapshot(lane,binding,plan['protocol'],pref),'lane':lane,'execution':execution,'boots':boot,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('ISSUE_MIX_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed={str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};names=('REFERENCE_CURSOR001.pt','REFERENCE_CURSOR002.pt')if lane=='reference'else('REPLAYED_CURSOR002.pt',)if lane=='cold'else('CURSOR001.pt','FINAL.pt','PASS1_LOGITS.npz','PASS2_LOGITS.npz');allowed.update(str(root/n)for n in names);sys.addaudithook(lambda e,a:development_io_audit(allowed,e,a));recovery_check_inputs(plan,physical=True);report=issue_mix_physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'ISSUE_MIX_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=issue_mix_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'ISSUE_MIX_BOOKEND_FAILED');result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_ISSUE_MIX_'+lane.upper()if error is None else'FAILED_CONSUMED','lane':lane,'plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal,'lane':lane}

def main_v3_issue_mix():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_issue_mix_execution
 p=argparse.ArgumentParser();p.add_argument('--v3-issue-mix-plan-sha256',required=True);p.add_argument('--lane',choices=ISSUE_MIX_LANES,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--v3-issue-mix-plan-sha256',a.v3_issue_mix_plan_sha256,'--lane',a.lane]+(['--preflight']if a.preflight else[]),'ISSUE_MIX_EXACT_CLI')
 if a.preflight:
  require_native_v3_issue_mix_execution(plan_sha256=a.v3_issue_mix_plan_sha256,lane=a.lane);plan,binding,pref,live=inspect_v3_issue_mix(a.v3_issue_mix_plan_sha256,a.lane,False);print(json.dumps({'status':'PREFLIGHT_V3_ISSUE_MIX_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'schedule_sha256':plan['schedule']['contract_sha256'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_issue_mix(a.v3_issue_mix_plan_sha256,a.lane),sort_keys=True))
# BEGIN GX1 V3 TRAIN EVAL GAP 20261128
TRAIN_EVAL_GAP_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001'
TRAIN_EVAL_GAP_STAGE=TRAIN_EVAL_GAP_NAME.lower()
TRAIN_EVAL_GAP_PLAN=REPO/'configs/research'/(TRAIN_EVAL_GAP_NAME+'.json')
TRAIN_EVAL_GAP_ROOT=Path('/home/andre2/GX1_RUNS')/TRAIN_EVAL_GAP_NAME
TRAIN_EVAL_GAP_ANCHOR='60b81b2535aea22dc7cd5ba3a08e3a9f2258a6d3'
TRAIN_EVAL_GAP_PROFILE='gx1_native_provider_hlc_v3_train_eval_gap_trainer20g_v1'
TRAIN_EVAL_GAP_DESIGN={'path':str(E/'V3_TRAIN_EVAL_GAP_DESIGN_20261128_003.json'),'sha256':'eb0f66e2e1454f3f0f56e10016be039d2cef56f421556a2363f5db461f34a000'}
TRAIN_EVAL_GAP_POLICY_SNAPSHOT=E/'V3_TRAIN_EVAL_GAP_ADMITTED_POLICY_20261128_001.json'
TRAIN_EVAL_GAP_PASSES=({'label':'EVAL_PARITY','training':False,'equal_tf':True},)+tuple({'label':'TRAIN_DRAW_'+str(i).zfill(2),'training':True,'equal_tf':False}for i in range(1,5))
TRAIN_EVAL_GAP_DISPATCH=b" if sys.argv[1:2]==['--v3-train-eval-gap-plan-sha256']:return main_v3_train_eval_gap()\n"

def train_eval_gap_reverse_legacy(raw):
 start=b'# BEGIN GX1 V3 TRAIN EVAL GAP 20261128\n';end=b'# END GX1 V3 TRAIN EVAL GAP 20261128\n'
 need(raw.count(start)==raw.count(end)and raw.count(start)<=1,'TRAIN_EVAL_GAP_ONE_BLOCK')
 if start in raw:
  first,tail=raw.split(start);_,last=tail.split(end);raw=first+last
 for fragment in (TRAIN_EVAL_GAP_DISPATCH,b' raw=train_eval_gap_reverse_legacy(raw)\n'):raw=raw.replace(fragment,b'')
 return raw

def train_eval_gap_source_bridge():
 raw=(REPO/SELF).read_bytes();need(raw.count(b'# BEGIN GX1 V3 TRAIN EVAL GAP 20261128\n')==1,'TRAIN_EVAL_GAP_ADDITIVE_BLOCK');old=subprocess.check_output(['git','show',TRAIN_EVAL_GAP_ANCHOR+':'+SELF],cwd=REPO);need(train_eval_gap_reverse_legacy(raw)==old,'TRAIN_EVAL_GAP_EXACT_ADDITIVE_REVERSAL')
 return {'producer_commit':TRAIN_EVAL_GAP_ANCHOR,'owner':SELF,'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':ref(REPO/SELF)['sha256'],'all_old_numeric_context_scoring_and_restore_owners_unchanged':True,'only_additive_train_eval_gap_dispatch_and_exact_bridge_normalization':True}

def train_eval_gap_gate(binding,admitted=False):
 return {'status':'ADMITTED_EXACT_V3_TRAIN_EVAL_GAP'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,'spending_authorized':False}

def train_eval_gap_require_gate(value,binding,admitted=False):
 expected=train_eval_gap_gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'TRAIN_EVAL_GAP_TYPED_ONE_USE_GATE');return value

def train_eval_gap_admission_snapshot(binding,policy_ref):
 sr=ref(TRAIN_EVAL_GAP_POLICY_SNAPSHOT);p=read(sr);need(policy_ref=={'path':str(POLICY),'sha256':sr['sha256']}and all(p.get(k)is False for k in DEV_FLAGS),'TRAIN_EVAL_GAP_SNAPSHOT_FLAGS_SHA');train_eval_gap_require_gate(p.get(TRAIN_EVAL_GAP_STAGE),binding,True);need(p[TRAIN_EVAL_GAP_STAGE].get('design')==TRAIN_EVAL_GAP_DESIGN,'TRAIN_EVAL_GAP_SNAPSHOT_DESIGN');return sr

def train_eval_gap_current_input_identity(producer):
 original,_,_,_,identity=recovery_original_metadata();need(original==producer['metadata']['original']and identity==producer['physical_input_identity'],'TRAIN_EVAL_GAP_FRESH_PHYSICAL_INPUT_CONTEXT');return identity

def train_eval_gap_final_metadata():
 design=read(TRAIN_EVAL_GAP_DESIGN);closed=issue_mix_closed('fit');producer=read(closed['plan']);original=producer['metadata']['original'];trajectory=producer['metadata']['trajectory'];report=closed['value']['report'];fr=report['final_checkpoint_receipt'];receipt=read(fr);saved=read(design['saved_eval_score'])
 need(closed['plan']==design['producer_fit_plan']and fr==design['frozen_FINAL_receipt']and receipt['checkpoint']==design['frozen_FINAL']and report['scores']['PASS2']==design['saved_eval_score']and saved['logits']==design['saved_eval_logits'],'TRAIN_EVAL_GAP_DESIGN_PRODUCER_OUTPUT_BINDINGS')
 need(fr['path']==str(ISSUE_MIX_ROOTS['fit']/'FINAL.json')and receipt['original_plan']==receipt['execution_plan']==closed['plan']and receipt['producer_lane']=='fit'and type(receipt['logical_cursor'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==432 and receipt['trajectory_sha256']==trajectory['contract_sha256']and receipt['checkpoint']['path']==str(ISSUE_MIX_ROOTS['fit']/'FINAL.pt'),'TRAIN_EVAL_GAP_AUTHENTIC_FINAL432')
 core.future_cuda_runtime_matches(producer['metadata']['runtime'],receipt['producer_runtime']);proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof['trajectory']==core.state_sha(trajectory)and proof['position']==core.state_sha(train.position(trajectory['schedule'],432))and proof['scores']==core.state_sha(report['scores'])and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None)and report['ending_cursor']==report['new_optimizer_updates']==432 and saved['rows']==20727 and saved['batches']==81,'TRAIN_EVAL_GAP_FINAL_SEMANTICS')
 budget=design['fixed_budget'];need(design['experiment_id']==TRAIN_EVAL_GAP_NAME and design['source_preparation_authorized']is True and design['execution_authorized']is design['training_authorized']is False and budget['NN_forwards']==405 and budget['eval_passes']==1 and budget['stochastic_train_no_grad_passes']==4 and budget['batch_rows']==256 and budget['batches_per_pass']==81 and budget['optimizer_updates']==budget['backward_calls']==budget['autograd_calls']==0 and design['cohort']['timeframes']==list(train.TF)and design['cohort']['years']==[2021,2022,2023,2024,2025]and design['cohort']['genuine_families']==8 and design['cohort']['D1_macro_preserved']is True,'TRAIN_EVAL_GAP_EXACT_FIXED_DESIGN')
 identity=train_eval_gap_current_input_identity(producer);refs={**copy.deepcopy(producer['inputs']),'design':TRAIN_EVAL_GAP_DESIGN,'final_execution_plan':closed['plan'],'final_result':closed['result'],'final_terminal':closed['terminal'],'final_claim':closed['claim'],'final_admitted_policy':closed['admitted_policy_snapshot'],'final_checkpoint_receipt':fr,'final_checkpoint':receipt['checkpoint'],'saved_eval_logits':design['saved_eval_logits'],'saved_eval_score':design['saved_eval_score'],'final_stream_binding':report['stream_binding'],**{'design_'+k:v for k,v in design['evidence'].items()}}
 return original,receipt,trajectory,producer,report['scores'],refs,identity

def build_v3_train_eval_gap_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_v3_train_eval_gap_source_bridge
 original,receipt,trajectory,producer,scores,inputs,identity=train_eval_gap_final_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'TRAIN_EVAL_GAP_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(trajectory['runtime_compatibility'],runtime)
 owners=dict(producer['source_owners']);owners.update({n:ref(REPO/n)['sha256']for n in(*GUARDS,SELF,DEV_TEST,DEV_GUARD_TEST)})
 for n,d in owners.items():need(ref(REPO/n)['sha256']==d,'TRAIN_EVAL_GAP_SOURCE:'+n)
 roles={k for k,v in inputs.items()if Path(v['path']).suffix not in('.pt','.npz','.npy')and not k.startswith('provider_raw_response_')};payload=set(inputs)-roles;limits=dict.fromkeys(producer['limits'],0);limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,target_archive_loads=1,checkpoint_loads=1,model_restores=1,model_forwards=405,backbone_forwards=405,readout_forwards=405,logit_archive_writes=5,logit_archive_readbacks=5,saved_eval_logit_loads=1)
 old={n:hashlib.sha256(subprocess.check_output(['git','show',TRAIN_EVAL_GAP_ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS};design=read(TRAIN_EVAL_GAP_DESIGN)
 return {'schema_version':'gx1_native_v3_train_eval_gap_plan_v1','stage':TRAIN_EVAL_GAP_STAGE,'source_commit':head,'output_root':str(TRAIN_EVAL_GAP_ROOT),'source_owners':owners,'guard_source_bridge':native_v3_train_eval_gap_source_bridge(old,producer_commit=TRAIN_EVAL_GAP_ANCHOR),'owner_source_bridge':train_eval_gap_source_bridge(),'design':TRAIN_EVAL_GAP_DESIGN,'original_plan':design['producer_fit_plan'],'checkpoint_receipt':inputs['final_checkpoint_receipt'],'metadata':{'original':original,'receipt':receipt,'trajectory':trajectory,'producer':producer,'scores':scores,'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':sorted(roles),'payload_stat':{k:{'size':Path(inputs[k]['path']).stat().st_size,'mtime_ns':Path(inputs[k]['path']).stat().st_mtime_ns}for k in sorted(payload)},'physical_input_identity':identity,'model_contract':original['model_contract'],'recipe':original['recipe'],'passes':list(TRAIN_EVAL_GAP_PASSES),'limits':limits,'execution_profile':TRAIN_EVAL_GAP_PROFILE,'execution':development_execution('warm',original),'scope':{k:design[k]for k in('question','single_intervention','measurement_scope','aggregation_contract','scientific_reporting','metric_availability')}}

def inspect_v3_train_eval_gap(sha,admitted=False):
 binding={'path':str(TRAIN_EVAL_GAP_PLAN),'sha256':sha};plan=read(binding);need(plan==build_v3_train_eval_gap_plan(plan['metadata']['runtime'],plan['source_commit']),'TRAIN_EVAL_GAP_EXACT_PLAN');recovery_check_inputs(plan);pref=ref(POLICY);p=read(pref);need(all(p.get(k)is False for k in DEV_FLAGS),'TRAIN_EVAL_GAP_ALL6_FLAGS_CLOSED');train_eval_gap_require_gate(p.get(TRAIN_EVAL_GAP_STAGE),binding,admitted);need(p[TRAIN_EVAL_GAP_STAGE].get('design')==TRAIN_EVAL_GAP_DESIGN,'TRAIN_EVAL_GAP_POLICY_DESIGN')
 if admitted:train_eval_gap_admission_snapshot(binding,pref)
 need(not TRAIN_EVAL_GAP_ROOT.exists()and not TRAIN_EVAL_GAP_ROOT.is_symlink(),'TRAIN_EVAL_GAP_FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def train_eval_gap_saved_outputs(binding,issues,counts):
 need(ref(binding['path'])==binding,'TRAIN_EVAL_GAP_SAVED_LOGITS_SHA')
 with np.load(binding['path'],allow_pickle=False)as saved:
  need(set(saved.files)=={'issue_ns','activity_logits','side_logits'}and saved['issue_ns'].dtype==issues.dtype and core.digest(saved['issue_ns'])==core.digest(issues),'TRAIN_EVAL_GAP_SAVED_ISSUE_BIJECTION');outputs={k:saved[k].copy()for k in('activity_logits','side_logits')}
 counts['saved_eval_logit_loads']+=1;need(all(v.shape==(20727,5)and v.dtype==np.float32 and np.isfinite(v).all()for v in outputs.values()),'TRAIN_EVAL_GAP_SAVED_F32_OUTPUTS');return outputs

def train_eval_gap_score(context,outputs):
 from gx1.audit import native_all_tf_provider_hlc_review_v3 as review
 # PASS2 selects the existing pure numeric owner; wrapper grants no candidate status.
 return review.review_cohort_outputs(truth=context['truth'],outputs=outputs,factory_prior=context['prior'],cohort_prior=context['scoring_prior'],selected_positions=context['positions'],source_rows=context['source_rows'],selection_ref=context['selection_ref'],spec=context['review_spec'],phase='PASS2',input_binding=context['input_binding'],normalization=context['normalization'])

def train_eval_gap_summary(reviews):
 need(len(reviews)==4,'TRAIN_EVAL_GAP_FOUR_FIXED_DRAWS')
 def metrics(review):
  result={'proper_partial_NLL':review['proper_partial_NLL']}
  def panel(path,value):
   for task,cell in value.items():
    for weight,model in (('row',cell['model']),('equal_target',(cell['equal_target_weight']or{}).get('model'))):
     for metric in ('NLL','brier','mean_positive_probability','mean_probability_minus_truth_pp','binary_argmax_positive_fraction','binary_argmax_negative_fraction','exact_tie_fraction'):
      if weight=='row'or metric in ('NLL','brier','mean_positive_probability'):result[path+'/'+task+'/'+weight+'/'+metric]=None if model is None else model.get(metric)
    result[path+'/'+task+'/row/AUC']=cell['rank_and_variation']['AUC']
    if 'calibration'in cell:
     for weight in ('row','equal_target'):
      calibration=cell['calibration'][weight];result[path+'/'+task+'/'+weight+'/calibration_mean_error_pp']=None if calibration is None else calibration['mean_probability_minus_truth_pp']
  for tf,value in review['all_TF'].items():panel('all_TF/'+tf,value)
  for tf,years in review['by_UTC_target_year'].items():
   for year,value in years.items():panel('by_UTC_target_year/'+tf+'/'+year,value)
  for tf,weights in review['equal_year_macro'].items():
   for weight,cell in weights.items():
    for metric in ('NLL','brier'):result['equal_year_macro/'+tf+'/'+weight+'/'+metric]=None if cell['model']is None else cell['model'][metric]
  return result
 values=[metrics(r)for r in reviews];need(all(set(v)==set(values[0])for v in values),'TRAIN_EVAL_GAP_SAME_METRIC_INVENTORY');result={}
 for path in values[0]:
  draws=[v[path]for v in values]
  if any(v is None for v in draws):result[path]={'status':'UNKNOWN','draw_values':draws};continue
  a=np.asarray(draws,np.float64);need(np.isfinite(a).all(),'TRAIN_EVAL_GAP_FINITE_SUMMARY');sd=float(a.std(ddof=1));result[path]={'status':'DESCRIPTIVE_ONLY','draw_values':draws,'mean':float(a.mean()),'sample_SD':sd,'descriptive_SE':sd/np.sqrt(len(a))}
 return {'draws':4,'metrics':result,'interpretation':'Mean/SD/SE of per-draw metrics, never score of averaged logits/probabilities. No confidence/significance/precision claim or best draw.','support_counts':'Preserved per draw; never averaged as performance metrics.','equal_target_AUC':'NOT_IMPLEMENTED','new_predictor':False}

def train_eval_gap_passes(model,optimizer,context,root,counts,steps,scores,proof,expected_proof,saved_outputs):
 import torch
 need(proof()==expected_proof and len(expected_proof)==10 and all(p.grad is None for p in model.parameters()),'TRAIN_EVAL_GAP_ORIGINAL_ALL10_EMPTY_GRAD')
 modes={n:m.training for n,m in model.named_modules()};rng=core.training_rng();rng_sha=core.state_sha(rng);immutable={k:expected_proof[k]for k in expected_proof if k not in('rng_state','training')};reports=[];reviews=[]
 def unchanged():
  actual=proof();return {**{k:actual[k]==v for k,v in immutable.items()},'grad_none':all(p.grad is None for p in model.parameters()),'named_module_inventory':set(dict(model.named_modules()))==set(modes)}
 try:
  for spec in TRAIN_EVAL_GAP_PASSES:
   model.train(spec['training']);need(all(m.training is spec['training']for _,m in model.named_modules()),'TRAIN_EVAL_GAP_ACTUAL_MODULE_MODES');before=core.state_sha(core.training_rng());pieces={};batches=core.all_score_batches(np.arange(20727,dtype=np.int64),256);need(len(batches)==81 and np.array_equal(np.concatenate(batches),np.arange(20727,dtype=np.int64)),'TRAIN_EVAL_GAP_CANONICAL_B256_BIJECTION')
   with torch.no_grad():
    for take in batches:
     packet=adapter.packet(context,take);output,_,_=core.future_cuda_forward(model,packet['features'],equal_tf=spec['equal_tf']);need(set(output)=={'activity_logits','side_logits'}and all(v.dtype==torch.float32 and tuple(v.shape)==(len(take),5)and not v.requires_grad for v in output.values()),'TRAIN_EVAL_GAP_NO_GRAD_F32_FORWARD')
     for key,value in output.items():pieces.setdefault(key,[]).append(value.detach().cpu().numpy().copy())
   outputs={k:np.concatenate(v)for k,v in pieces.items()};need(all(np.isfinite(v).all()for v in outputs.values()),'TRAIN_EVAL_GAP_FINITE_OUTPUTS');after=core.state_sha(core.training_rng());observed=unchanged();need(all(observed.values()),'TRAIN_EVAL_GAP_PRECLEANUP_MUTATION')
   if not spec['training']:
    need(before==after==rng_sha and all(core.digest(outputs[k])==core.digest(saved_outputs[k])for k in outputs),'TRAIN_EVAL_GAP_EXACT_SAVED_EVAL_PARITY')
   else:
    need(before!=after and before==reports[-1]['RNG_after_sha256']and (len(reports)>1 or before==rng_sha),'TRAIN_EVAL_GAP_SEQUENTIAL_RNG')
   review=train_eval_gap_score(context,outputs);need(core.state_sha(core.training_rng())==after,'TRAIN_EVAL_GAP_SCORING_RNG_UNCHANGED');logits_ref=development_logits(root,spec['label'],context['truth']['issue_ns'],outputs,counts);rr=development_publish(root,spec['label']+'_SCORE.json',{'label':spec['label'],'execution_mode':spec,'RNG_before_sha256':before,'RNG_after_sha256':after,'rows':20727,'batches':81,'truth_sha256':context['truth_sha256'],'source_rows_sha256':core.digest(context['source_rows']),'issue_ns_sha256':core.digest(context['truth']['issue_ns']),'logits':logits_ref,'review':review,'numeric_owner_phase':'PASS2','diagnostic_only':True,'no_new_trained_candidate':True,'no_draw_selection':True,'equal_target_AUC':'NOT_IMPLEMENTED','new_candidate':False,'observed_immutable_state_before_cleanup':observed});reports.append({'label':spec['label'],'score':rr,'logits':logits_ref,'RNG_before_sha256':before,'RNG_after_sha256':after})
   if spec['training']:reviews.append(review)
  observed=unchanged();need(all(observed.values()),'TRAIN_EVAL_GAP_PRECLEANUP_MUTATION');summary=development_publish(root,'DRAW_METRIC_SUMMARY.json',train_eval_gap_summary(reviews));return {'passes':reports,'summary':summary,'observed_immutable_state_before_cleanup':observed,'exact_saved_EVAL_F32_parity':True,'RNG_restored_only_after_all_four_draws':True,'named_module_modes_sha256':core.state_sha(modes),'new_predictor':False,'diagnostic_only':True,'no_new_trained_candidate':True,'no_draw_selection':True,'equal_target_AUC':'NOT_IMPLEMENTED'}
 finally:
  # Never reload weights, buffers or optimizer to conceal a mutation.
  observed=unchanged()
  for name,module in model.named_modules():
   if name in modes:module.training=modes[name]
  core.restore_training_rng(rng);final_proof=proof();final_modes={n:m.training for n,m in model.named_modules()};restored=all(observed.values())and final_modes==modes and core.state_sha(core.training_rng())==rng_sha and final_proof==expected_proof and all(p.grad is None for p in model.parameters())
  development_publish(root,'STATE_INVARIANTS.json',{'counts':dict(counts),'observed_immutable_state_before_cleanup':observed,'all10_semantic_proofs_unchanged':final_proof==expected_proof,'final_full10_proof':final_proof,'named_module_modes_restored':final_modes==modes,'full_RNG_restored':core.state_sha(core.training_rng())==rng_sha,'only_module_modes_and_RNG_restored':True,'completed_passes':reports,'status':'PASS'if restored else'FAILED_CONSUMED'})
  need(restored,'TRAIN_EVAL_GAP_PRECLEANUP_OR_FINAL_ALL10_MUTATION')

def train_eval_gap_physical(plan,binding,root,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 original=plan['metadata']['original'];receipt=plan['metadata']['receipt'];trajectory=plan['metadata']['trajectory'];counts=dict.fromkeys(plan['limits'],0);context=development_nn_context(original);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;producer=plan['metadata']['producer'];stream=issue_mix_actual_stream(context,producer);need(read(plan['inputs']['final_stream_binding'])==stream,'TRAIN_EVAL_GAP_AUTHENTIC_SOURCE_ISSUE_ALL5_MASK_STREAM');core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'TRAIN_EVAL_GAP_AUTHENTIC_FACTORY');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'TRAIN_EVAL_GAP_FRESH_ADAMW');need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'TRAIN_EVAL_GAP_FINAL_SHA_BEFORE_LOAD');cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 need(cp['producer']=={'original_plan':plan['original_plan'],'lane':'fit','runtime':receipt['producer_runtime']}and cp['scores']==plan['metadata']['scores'],'TRAIN_EVAL_GAP_ACTUAL_PRODUCER_SCORES');issue_mix_validate_checkpoint(cp,producer,plan['original_plan'],432,stream);restored=train.strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=plan['original_plan'],expected_cursor=432,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;need(restored==train.position(trajectory['schedule'],432),'TRAIN_EVAL_GAP_RESTORED_FINAL432');steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
 def proof():return train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan['original_plan'],lane='diagnostic',cursor=432,steps=steps,scores=scores,producer_runtime=live))
 saved=train_eval_gap_saved_outputs(plan['inputs']['saved_eval_logits'],context['truth']['issue_ns'],counts)
 def counter(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'TRAIN_EVAL_GAP_FORWARD_BOUND')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 try:
  report=train_eval_gap_passes(model,optimizer,context,root,counts,steps,scores,proof,receipt['proof'],saved);need(counts==plan['limits']and proof()==receipt['proof']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'TRAIN_EVAL_GAP_EXACT_COUNTS_ALL10_VRAM');return {**report,'counts':counts,'final_checkpoint_receipt':plan['checkpoint_receipt'],'final_state_proof':receipt['proof'],'all10_semantic_proofs_unchanged':True,'optimizer_updates':0,'new_checkpoint_writes':0,'original_FINAL_never_rewritten':True,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN','scientific_limits':plan['scope']}
 finally:
  for hook in hooks:hook.remove()

def train_eval_gap_bookends(plan,binding,pref,boot):
 checks={};tests={'plan':lambda:ref(TRAIN_EVAL_GAP_PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==d for n,d in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and train_eval_gap_admission_snapshot(binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boot,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']}
 for name,fn in tests.items():
  try:checks[name]=bool(fn())
  except BaseException:checks[name]=False
 return checks

def train_eval_gap_payload_allowlist(plan,root):
 inputs={str(Path(v['path']).resolve())for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')}
 return inputs|{str((root/(spec['label']+'_LOGITS.npz')).resolve())for spec in TRAIN_EVAL_GAP_PASSES}

def run_v3_train_eval_gap(sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_train_eval_gap_execution
 execution=require_native_v3_train_eval_gap_execution(plan_sha256=sha);plan,binding,pref,live=inspect_v3_train_eval_gap(sha,True);root=TRAIN_EVAL_GAP_ROOT;root.mkdir(exist_ok=False);boot={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':train_eval_gap_admission_snapshot(binding,pref),'execution':execution,'boots':boot,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
 def stop(*_):raise TimeoutError('TRAIN_EVAL_GAP_INTERNAL_WALL_OR_SIGNAL')
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
 try:
  allowed=train_eval_gap_payload_allowlist(plan,root);sys.addaudithook(lambda e,a:development_io_audit(allowed,e,a));recovery_check_inputs(plan,physical=True);report=train_eval_gap_physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'TRAIN_EVAL_GAP_ZERO_FIREWALL_DENIALS')
 except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);book=train_eval_gap_bookends(plan,binding,pref,boot);error=error or(None if all(book.values())else'TRAIN_EVAL_GAP_BOOKEND_FAILED');preserved={p.name:ref(p)for p in sorted(root.iterdir())if p.is_file()and p.name!='CLAIM.json'};result=development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_TRAIN_EVAL_GAP'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0,'new_candidates':0,'preserved_measurements':preserved,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});terminal=development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False})
  for sig,handler in handlers.items():signal.signal(sig,handler)
 if error:raise RuntimeError(error)
 return {'result':result,'terminal':terminal}

def main_v3_train_eval_gap():
 from gx1.contracts.gx1_capped_execution_v1 import require_native_v3_train_eval_gap_execution
 parser=argparse.ArgumentParser();parser.add_argument('--v3-train-eval-gap-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();sha=a.v3_train_eval_gap_plan_sha256;need(sys.argv[1:]==['--v3-train-eval-gap-plan-sha256',sha]+(['--preflight']if a.preflight else[]),'TRAIN_EVAL_GAP_EXACT_CLI')
 if a.preflight:
  require_native_v3_train_eval_gap_execution(plan_sha256=sha);plan,binding,pref,live=inspect_v3_train_eval_gap(sha,False);print(json.dumps({'status':'PREFLIGHT_V3_TRAIN_EVAL_GAP_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'final_checkpoint_receipt':plan['checkpoint_receipt'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_v3_train_eval_gap(sha),sort_keys=True))
# END GX1 V3 TRAIN EVAL GAP 20261128
# END GX1 V3 DIRECTION ISSUE MIX 20261128
# END GX1 V3 BLOCK ORDER RECOVERY 20261128
# END GX1 V3 DIRECTION BLOCK ORDER 20261128
# END GX1 V3 DIRECTION GRADIENT 20261128
# END GX1 V3 EXACT RECOVERY 20261128
# END GX1 V3 BOUNDED DEVELOPMENT 20261127
if __name__=='__main__':main()
